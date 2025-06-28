import os
import regex as re
from typing import BinaryIO
from typing import Iterable, Iterator
from collections import defaultdict
from multiprocessing import Process, Queue
import time
import tdqm


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]| None = None
):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []