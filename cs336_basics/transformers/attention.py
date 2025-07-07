import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
from .rope import RotaryPositionalEmbedding
from .linear import Linear
from .utils import softmax
import einx



def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Implement scaled dot-product attention.

    Args:
        query (Float[Tensor, "batch_size ... seq_len d_k"]): Query tensor.
        key (Float[Tensor, "batch_size ... seq_len d_k"]): Key tensor.
        value (Float[Tensor, "batch_size ... seq_len d_v"]): Value tensor.
        mask (Bool[Tensor, "seq_len seq_len"]): Optional boolean mask.

    Returns:
        Float[Tensor, "batch_size ... seq_len d_v"]): Output tensor.
    """
    d_k = key.shape[-1]
    scores = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k) 

    if mask is not None:
        scores = scores.masked_fill(mask == False, -torch.inf)

    attention_probs = softmax(scores, dim=-1)
    output = attention_probs @ value
    return output

class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention with causal masking and RoPE.
    """

    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 use_rope: bool = True, 
                 max_seq_len: int = 2048, 
                 theta: int = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_o = Linear(d_model, d_model)

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_rope = use_rope

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(self.theta, self.d_head, self.max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor  | None = None) -> torch.Tensor:
        """
        Args:
            x (Float[Tensor, "batch_size seq_len d_model"]): Input tensor.
            token_positions: The positional indices along the sequence dimension of the input embeddings.
        
        Returns:
            Float[Tensor, "batch_size seq_len d_model"]): Output tensor.
        """
        *b, sequence_length, d_model = x.size()
        qx,kx,vx = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        # Reshape to split into heads (batch, seq_len, d_model)
        qx = rearrange(qx, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        kx = rearrange(kx, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        vx = rearrange(vx, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        
        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

        # Duplicate token positions for each head
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        if self.use_rope:
            qx = self.rope(qx, token_positions)
            kx = self.rope(kx, token_positions)
        
        # # Create causal mask
        seq_len = x.shape[1]
        own_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1).unsqueeze(0).unsqueeze(0)
        causal_mask = ~own_mask
        
        attn_output = scaled_dot_product_attention(query = qx, key = kx, value = vx, mask=causal_mask)
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        output = self.linear_o(attn_output)
        return output