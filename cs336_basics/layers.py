import torch
from torch.nn import Module, Parameter
from torch.nn.init import trunc_normal_
from einops import einsum


class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """_summary_

        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        weight = torch.zeros(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** (1/2)
        self.weight = Parameter(
            trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out_f in_f, ... in_f -> ... out_f")
