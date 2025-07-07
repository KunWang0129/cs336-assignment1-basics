import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union, BinaryIO, IO


def data_loader(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    vocab_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a batch of input and target sequences from the tokenized data.

    This function is designed to work with large datasets by supporting memory-mapped
    numpy arrays.

    Args:
        x: A numpy array of token IDs. Can be a memory-mapped array.
        batch_size: The number of sequences in a batch.
        context_length: The length of each sequence.
        device: The PyTorch device to place the tensors on.
        vocab_size: The size of the vocabulary. If provided, the sampled token IDs
                    are validated to be within the range [0, vocab_size).

    Returns:
        A tuple of two tensors:
        - The input sequences, with shape (batch_size, context_length).
        - The target sequences, with shape (batch_size, context_length).
    """
    n = len(x)

    # The last possible starting index for an input sequence is n - context_length - 1
    if n <= context_length:
        raise ValueError("Length of data array must be greater than context_length.")

    max_start_index = n - context_length - 1

    # We sample from [0, max_start_index]
    start_indices = np.random.randint(0, max_start_index + 1, size=batch_size)

    # Create the input and target sequences using list comprehensions
    input_sequences = [x[i : i + context_length] for i in start_indices]
    target_sequences = [x[i + 1 : i + context_length + 1] for i in start_indices]

    # Convert lists of numpy arrays to a single numpy array for validation and tensor conversion
    inputs_np = np.array(input_sequences)
    targets_np = np.array(target_sequences)

    # Validate token IDs if vocab_size is provided
    if vocab_size is not None:
        if not (np.all(inputs_np >= 0) and np.all(inputs_np < vocab_size)):
            raise ValueError(f"Input token IDs are out of vocabulary range [0, {vocab_size}).")
        if not (np.all(targets_np >= 0) and np.all(targets_np < vocab_size)):
            raise ValueError(f"Target token IDs are out of vocabulary range [0, {vocab_size}).")

    # Convert numpy arrays to PyTorch tensors
    inputs = torch.tensor(inputs_np, dtype=torch.long)
    targets = torch.tensor(targets_np, dtype=torch.long)

    # Move tensors to the specified device
    if "cuda" in device:
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs = inputs.to(device)
        targets = targets.to(device)

    return inputs, targets

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
):
    """
    Saves a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: The current training iteration.
        out: The path or file-like object to save the checkpoint to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    """
    Loads a training checkpoint.

    Args:
        src: The path or file-like object to load the checkpoint from.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        The iteration number from the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
