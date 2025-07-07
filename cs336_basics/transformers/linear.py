import math
import torch
from einops import einsum

class Linear(torch.nn.Module):
    """
    A linear transformation module.

    This module applies a linear transformation to the incoming data: y = xA^T.
    It is similar to `torch.nn.Linear` but does not include a bias term.
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """
        Constructs a linear transformation module.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            device: The desired device of the parameters.
            dtype: The desired data type of the parameters.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(self.init_parameters())
        

    def init_parameters(self) -> None:
        """
        Initialize the weight parameter.
        """
        weight = torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype)
        std = math.sqrt(2.0/(self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a = -3*std, b = 3*std)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation.

        Args:
            x: The input tensor of shape (*, in_features).

        Returns:
            The output tensor of shape (*, out_features).
        """
        
        return einsum(x, self.weight, "... in, out in -> ... out")
