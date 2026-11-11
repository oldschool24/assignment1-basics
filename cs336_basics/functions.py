import torch


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """Apply softmax to the 𝑖-th dimension of the input tensor"""

    # to avoid numerical stability issues:
    # substract max value in i-th dimension from all elements
    max_logits = torch.max(x, dim=i, keepdim=True).values
    shifted = x - max_logits

    exps = torch.exp(shifted)
    denominator = exps.sum(dim=i, keepdim=True)
    result = exps / denominator

    return result
