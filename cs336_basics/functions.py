import torch
from einops import einsum
from math import sqrt


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

def scaled_dot_product_attention(queries: torch.Tensor,
                                 keys: torch.Tensor,
                                 values: torch.Tensor,
                                 mask: torch.Tensor | None = None):
    """ Attention operation = softmax(Q * K_t / sqrt(d_k)) * V

    Args:
        queries (torch.Tensor): (batch_size, ..., seq_len, d_k)
        keys (torch.Tensor): (batch_size, ..., seq_len, d_k)
        values (torch.Tensor): (batch_size, ..., seq_len, d_v)
        mask (torch.Tensor): true at position (𝑖, 𝑗) --
        indicates that the query 𝑖 does attend to the key 𝑗
    """
    d_k = keys.shape[-1]
    product = einsum(queries, keys, "... seq1 d, ... seq2 d -> ... seq1 seq2") # Q * K_transposed
    logits = product / sqrt(d_k)
    if mask is not None:
        logits.masked_fill_(~mask, -torch.inf)
    attn = softmax(logits, -1)
    result = einsum(attn, values, "... seq1 seq2, ... seq2 d -> ... seq1 d")
    return result
