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
    context_length = 256
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
    train_bpe_TinyStories(trainfile_name, vocab_size, special_tokens, vocab_name, merges_name)
    Tokenizer_TinyStories(trainfile_name, validfile_name, trainencode_name, validencode_name, vocab_name, merges_name, special_tokens)
    
