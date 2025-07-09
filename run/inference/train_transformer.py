import sys
import os
from tqdm import tqdm
import wandb
import argparse
import torch
import time
import pickle
import pathlib
import numpy as np
from typing import Iterable

# Add project root to python path to allow importing from cs336_basics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cs336_basics.bpe_token.trainer import train_bpe
from cs336_basics.bpe_token.tokenizer import BPETokenizer
from cs336_basics.transformers.transformer import TransformerLM
from cs336_basics.transformers.optimizer import AdamW, get_cosine_lr
from cs336_basics.transformers.utils import cross_entropy, clip_gradient
from cs336_basics.transformers.training import data_loader, save_checkpoint, load_checkpoint

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data"
MODULE_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "module"

def save_pkl(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
        return file

def save_encode(file, file_name):
    np.array(file, dtype=np.uint16).tofile(file_name)

def save_encode_stream(token_stream: Iterable[int], file_path: os.PathLike):
    array = np.fromiter(token_stream, dtype=np.uint16)
    array.tofile(file_path)

def train_bpe_TinyStories(
    file_name: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    vocab_name: str, 
    merges_name: str
):
    start_time = time.time()
    traindata_path = DATA_PATH / file_name
    vocab, merges = train_bpe(
        input_path=traindata_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    save_pkl(vocab, DATA_PATH / vocab_name)
    save_pkl(merges, DATA_PATH / merges_name)
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"执行时间: {minutes} 分 {seconds} 秒")

def Tokenizer_TinyStories(
    trainfile_name: str | os.PathLike, 
    validfile_name: str | os.PathLike, 
    trainencode_name: str | os.PathLike, 
    validencode_name: str | os.PathLike, 
    vocab_name: str | os.PathLike, 
    merges_name: str | os.PathLike, 
    special_tokens: list[str]
):
    start_time = time.time()
    trainfile_path = DATA_PATH / trainfile_name
    validfile_path = DATA_PATH / validfile_name
    trainencode_path = DATA_PATH / trainencode_name
    validencode_path = DATA_PATH / validencode_name
    tokenizer = BPETokenizer()
    tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)

    with open(trainfile_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    encode_stream = tokenizer.encode_iterable(train_lines)
    save_encode_stream(encode_stream, trainencode_path)

    with open(validfile_path, 'r', encoding='utf-8') as f:
        valid_lines = f.readlines()
    encode_stream = tokenizer.encode_iterable(valid_lines)
    save_encode_stream(encode_stream, validencode_path)


@torch.no_grad()
def evaluate_validloss(model, valid_dataset, batch_size, context_length, device):
    model.eval()
    losses = []
    total_batches = len(valid_dataset) // (batch_size * context_length)

    for i in range(total_batches):
        input_batch, target_batch = data_loader(valid_dataset, batch_size, context_length, device)
        logits = model(input_batch)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def generate_sample_and_log(model, tokenizer, prompt_str, device, iteration, max_gen_tokens=256, temperature=1.0, top_p=0.95):
    model.eval()
    with torch.no_grad():
        prompt_ids = tokenizer.encode(prompt_str)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        eos_token_id = tokenizer.vocab_to_id.get("<|endoftext|>".encode('utf-8'), None)

        gen_ids = model.decode(
            input_tensor,
            max_new_tokens=max_gen_tokens,
            temperature=temperature,
            top_p=top_p,
            end_of_sequence_token_id=eos_token_id,
        )

        full_ids = prompt_ids + gen_ids[0].tolist()
        output_text = tokenizer.decode(full_ids)

        print(f"[Sample @ Iter {iteration}] {output_text}")
        wandb.log({"sample/text": wandb.Html(f"<pre>{output_text}</pre>")})

    model.train()

if __name__ == '__main__':
    trainfile_name = 'TinyStoriesV2-GPT4-valid.txt'
    validfile_name = 'TinyStoriesV2-GPT4-valid.txt'
    vocab_name = 'TinyStoriesV2-GPT4_vocab.pkl'
    merges_name = 'TinyStoriesV2-GPT4_merges.pkl'
    trainencode_name = 'TStrain_tokens.bin'
    validencode_name = 'TSvalid_tokens.bin'
    vocab_size = 10000
    batch_size = 128
    context_length = 128 # 256
    d_model = 512
    d_ff = 1344
    initial_lr = 0.0033
    lr = 0.0033
    rope_theta = 10000
    n_layers = 4
    n_heads = 16
    max_l2_norm = 1e-2
    max_gen_tokens = 256
    temperature = 0.8
    top_p = 0.95
    special_tokens = ["<|endoftext|>"]
    # BPE Tokenizers
    tokenizer = BPETokenizer()
    tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)    
    # Train
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    train_dataset = np.memmap(DATA_PATH / trainencode_name, dtype=np.uint16, mode="r")
    valid_dataset = np.memmap(DATA_PATH / validencode_name, dtype=np.uint16, mode="r")
    start_iter = 0
    total_iters = 1200
    log_interval = total_iters // 200
    ckpt_interval = total_iters // 20
    print(f"Total iterations: {total_iters}")
    # init wandb
    wandb.init(
        project="cs336_assignment1",
        name=f"run-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "batch_size": batch_size,
            "context_length": context_length,
            "max_lr": lr,
            "min_lr": max(1e-6, lr * 0.01),
            "warmup_iters": min(500, total_iters*0.1),
            "cosine_iters": total_iters,
        }
    )
    # model
    model = TransformerLM(
        vocab_size=vocab_size, 
        context_length=context_length, 
        num_layers=n_layers, 
        d_model=d_model, 
        num_heads=n_heads, 
        d_ff=d_ff, 
        rope_theta=rope_theta
    ).to(device)
    # AdamW use default lr, betas, eps, weight_decay
    optimizer = AdamW(model.parameters(), lr=lr)
    # Resume checkpoint
    ckpt_path = MODULE_PATH / 'TScheckpoint.pt'
    if ckpt_path.exists():
        start_iter = load_checkpoint(src=ckpt_path, model=model, optimizer=optimizer)
    model.train()
    wandb.watch(model, log="all")
    pbar = tqdm(total=total_iters)
    iteration = start_iter
    best_val_loss = float('inf')
    val_interval = total_iters // 20
    while iteration < total_iters:
        input_train, target_train = data_loader(train_dataset, batch_size, context_length, device)
        logits = model(input_train)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), target_train.view(-1))
        lr = get_cosine_lr(
            iteration,
            max_learning_rate=initial_lr,
            min_learning_rate=max(1e-6, initial_lr * 0.01),
            warmup_iters=int(min(500, total_iters * 0.1)),
            cosine_cycle_iters=total_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(model.parameters(), max_l2_norm)
        optimizer.step()
        if iteration % log_interval == 0:
            print(f"[Iter {iteration}] loss: {loss.item():.4f}")
            wandb.log({"train/loss": loss.item(), "lr": lr}, step=iteration)
        if iteration % ckpt_interval == 0:
            save_checkpoint(model, optimizer, iteration, ckpt_path)
        if iteration % val_interval == 0:
            val_loss = evaluate_validloss(model, valid_dataset, batch_size, context_length, device)
            print(f"[Iter {iteration}] Validation loss: {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=iteration)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iteration, MODULE_PATH / 'TS_best_checkpoint.pt')
                print(f"Saved best model (val_loss={val_loss:.4f})")
                wandb.run.summary["best_val_loss"] = best_val_loss
            # optional: generate sample after validation
            generate_sample_and_log(model=model,
                tokenizer=tokenizer,
                prompt_str="Once upon a time", 
                device=device,
                iteration=iteration,
                max_gen_tokens=max_gen_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        iteration += 1
        pbar.update(1)
    wandb.finish()
