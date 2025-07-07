import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .utils import softmax
from .positionwiseff import PositionwiseFeedForward
from .rmsnorm import RMSNorm
from .embedding import Embedding
from .linear import Linear
from .rope import RotaryPositionalEmbedding

class TransformerBlock(nn.Module):
    """
    A single Transformer block, as described in the paper.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 2048, rope_theta: float = 10000.0):
        """
        Initializes the Transformer block.

        Args:
            d_model: The dimensionality of the input and output.
            num_heads: The number of attention heads.
            d_ff: The dimensionality of the inner layer of the feed-forward network.
            rope_theta: The theta parameter for RoPE.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta=rope_theta, max_seq_len=max_seq_len)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Transformer block.

        Args:
            x: The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            The output tensor of shape (batch_size, seq_len, d_model).
        """
        # Attention sub-layer with pre-normalization
        x_attn = self.attention(self.norm1(x))
        attn_output = x + x_attn

        # Feed-forward sub-layer with pre-normalization
        ff_output = self.feed_forward(self.norm2(attn_output))
        output = attn_output + ff_output

        return output


class TransformerLM(nn.Module):
    """
    A Transformer language model.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.d_head = d_model // num_heads
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Transformer language model.

        Args:
            token_ids: The input tensor of shape (batch_size, seq_len).

        Returns:
            The output tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = token_ids.shape
        
        # Token and position embeddings
        x = self.token_embeddings(token_ids)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization and linear layer
        x = self.ln_final(x)
        return self.lm_head(x)
        
    def decode(self, prompt: torch.Tensor, 
               max_new_tokens: int, 
               temperature: float = 1.0, 
               top_p: float = 1.0, 
               end_of_sequence_token_id: int = -1
               ):
        """
        Generates text from the model.

        Args:
            prompt: The input tensor of shape (batch_size, seq_len).
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature for scaling the logits.
            top_p: The nucleus sampling probability.
            end_of_sequence_token_id: The token ID that marks the end of a sequence.

        Returns:
            The generated sequence of token IDs.
        """
        
        generated = prompt
        for _ in range(max_new_tokens):
            
            logits = self.forward(generated)[:, -1, :]
            
            # Temperature scaling
            logits = logits / temperature
            
            # Softmax to get probabilities
            probs = softmax(logits, dim=-1)
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find indices to remove
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[:, indices_to_remove] = 0
                
                # Renormalize
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token
            generated = torch.cat((generated, next_token), dim=1)
            
            # Check for end of sequence
            if next_token.item() == end_of_sequence_token_id:
                break
                
        return generated