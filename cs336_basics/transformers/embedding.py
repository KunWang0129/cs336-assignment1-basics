import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    A simple embedding layer.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Constructs an embedding module.

        Args:
            num_embeddings: Size of the vocabulary.
            embedding_dim: Dimension of the embedding vectors.
            device: Device to store the parameters on.
            dtype: Data type of the parameters.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Looks up the embedding vectors for the given token IDs.

        Args:
            token_ids: A tensor of token IDs.

        Returns:
            A tensor of embedding vectors.
        """
        batch_size, sequence_length = token_ids.shape
        output = torch.empty(batch_size, sequence_length, self.embedding_dim)

        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                output[i][j] = self.weight[token_id]
        return output
