import torch
from torch.nn import Module, Parameter
from torch.nn.init import trunc_normal_
from einops import einsum


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """Construct a linear transformation module.

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


class Embedding(Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., 𝑑_model
            device (_type_, optional): Device to store the parameters on. Defaults to None.
            dtype (_type_, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        weight = torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = Parameter(
            trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""

        # The forward method should select the embedding vector for each token ID
        # by indexing into an embedding matrix of shape (vocab_size, d_model) using a
        # torch.LongTensor of token IDs with shape (batch_size, sequence_length).

        return self.weight[token_ids, :]

