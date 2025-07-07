import os
import regex as re
from typing import BinaryIO
from typing import Iterable, Iterator
from collections import defaultdict
from multiprocessing import Process, Queue
import time
import tqdm
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cs336_basics.bpe_token.trainer import pretokenize, train_bpe


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]| None = None
):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    def from_files(cls, vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None
    ):
        raise NotImplementedError
    
    def encode(self, text:str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        byte_pretokens = pretokenize(text, self.special_tokens, drop_special_token=False)
        byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        pretokens = [] # type: list[list[int]]

        # reverse vocab for byte to list[int]
        vocab_reversed = {v: k for k, v in self.vocab.items()}

        # Convert pretokens fro bytes to list[int] by vocab
        for i, pretoken in enumerate(byte_pretokens):
            new_pretoken = []

            if pretoken in byte_special_tokens:
                new_pretoken.append(vocab_reversed[pretoken])
            else:
                for byte in pretoken:
                    new_pretoken.append(vocab_reversed[bytes([byte])])
            
            pretokens.append(new_pretoken)
        
        # Apply merges
        for i, pretoken in enumerate(pretokens):
            # Iterate through merges
            for merge in self.merges:
                new_pretoken = []
                new_index = vocab_reversed[merge[0] + merge[1]]
                j = 0
                # Iterate through pretoken and apply merge
                while j < len(pretoken):
                    if (j < len(pretoken)-1) and ((self.vocab[pretoken[j]], self.vocab[pretoken[j+1]]) == merge):
                        new_pretoken.append(new_index)
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1
                pretoken = new_pretoken
            pretokens[i] = pretoken

        tokens = [token for pretoken in pretokens for token in pretoken]
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files 
        that we cannot directly load into memory.
        """
        for text in iterable:
            tokens = self.encode(text)
            for idx in tokens:
                yield idx
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a string.
        """
        tokens = bytes()
        replacemenc_char = b'\uFFFD'  # Unicode replacement character
        vocab_size = len(self.vocab)

        for token_id in token_ids:
            if token_id < vocab_size:
                # If index is within bounds, append the corresponding byte
                token = self.vocab[token_id]
            else:
                # If index is out of bounds, use replacement character
                token = bytes(replacemenc_char, encoding='utf-8')
            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')
        return decoded
    
def main():
    file_path = 'tests/fixtures/tinystories_sample.txt'
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    # Train BPE
    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    tokenizer = BPETokenizer(vocab, merges, special_tokens)

    # Encode a sample text
    sample_text = "Hello, world! <|endoftext|> This is a test. <|endoftext|> Let's see how it works.<|endoftext|>"
    ebcoded_tokens = tokenizer.encode(sample_text)
    print("Encoded tokens:", ebcoded_tokens)
    decoded = [tokenizer.decode([x]) for x in ebcoded_tokens]
    print("Decoded text:", decoded)

    print(sample_text == decoded) # Decoded text does not match the original text.

if __name__ == "__main__":
    main()