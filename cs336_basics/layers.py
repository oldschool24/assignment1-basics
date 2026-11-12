import torch
from torch.nn import Module, Parameter
from torch.nn.init import trunc_normal_
from einops import einsum, reduce, rearrange
from .functions import SiLU, scaled_dot_product_attention


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


class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """root mean square layer normalization

        Args:
            d_model (int): Hidden dimension of the model
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (_type_, optional): Device to store the parameters on. Defaults to None.
            dtype (_type_, optional): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        self.weight = Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(reduce(x ** 2, '... d_model -> ... 1', 'mean') + self.eps)
        result = (x / rms) * self.weight

        return result.to(dtype=in_dtype)


class FFNN(Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None, func='SwiGLU'):
        """Feed-forward network with SwiGLU or SiLU as activation function.

        You should set 𝑑ff to approximately 8/3 × 𝑑_model in your implementation (SwiGLU),
        while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64
        to make good use of your hardware.
        """
        super().__init__()
        self.func = func
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        if func == 'SwiGLU':
            self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x1 = SiLU(x1)
        if self.func == 'SwiGLU':
            x1 = x1 * self.w3(x)   # GLU
        result = self.w2(x1)
        return result


class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the RoPE module and create buffers if needed.

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be input
            device (_type_, optional): Device to store the buffer on. Defaults to None.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_length = max_seq_len

        i = torch.arange(0, max_seq_len, device=device).unsqueeze(-1)
        k = torch.arange(1, d_k//2 + 1, device=device).unsqueeze(0)
        theta_ik = i / (theta ** ((2*k - 2)/d_k))
        sin = torch.sin(theta_ik)
        cos = torch.cos(theta_ik)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply ROPE to input vector x.

        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): tensor of shape (..., seq_len) specifying the token positions of 𝑥 along the sequence dimension

        Returns:
            torch.Tensor: same shape as x
        """

        # use the token positions to slice cos and sin tensors along the sequence dimension
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_odd = x[..., 0::2]
        x_even = x[..., 1::2]

        new_x_odd = cos * x_odd - sin * x_even
        new_x_even = sin * x_odd + cos * x_even
        result = rearrange([new_x_odd, new_x_even], "h ... num_pairs -> ... (num_pairs h)")

        return result


class MHSA(Module):
    def __init__(self, d_model: int, num_heads: int, rope=None,
                 theta=10000, max_seq_len=256, device=None, dtype=None):
        """Causal multi-head self-attention.

        Args:
            d_model (int): Dimensionality of the Transformer block inputs
            num_heads (int): Number of heads to use in multi-head self-attention
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v, device, dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device, dtype)
        if rope is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        """Implementation of multi-head attention.

        Args:
            x (torch.Tensor, "... seq_len d_model"): input tensor
            token_positions (torch.Tensor): tensor of shape (..., seq_len)
            specifying the token positions of 𝑥 along the sequence dimension

        Returns:
            torch.Tensor: output of multi-head self attention
        """
        # 1. computing the key, value, and query projection
        queries = self.q_proj(x)  # (..., seq, h*d_k)
        keys = self.k_proj(x)     # same
        values = self.v_proj(x)   # (..., seq, h*d_v)
        queries = rearrange(queries, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        keys = rearrange(keys, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        values = rearrange(values, "... seq (h d_v) -> ... h seq d_v", h=self.num_heads)
        seq_len = queries.shape[-2]  # how many tokens in input

        # 2. Rope
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                token_positions = rearrange(token_positions, "seq -> 1 seq")
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        # 3. Causal masking
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)   # lower-triangular

        # 4. apply self-attention
        multihead = scaled_dot_product_attention(queries, keys, values, mask)

        # 5. project back
        multihead = rearrange(multihead, "... h seq d_v -> ... seq (h d_v)", h=self.num_heads)
        result = self.output_proj(multihead)

        return result


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope=None,
                 max_seq_len=256, theta=10000, device=None):
        """Pre-norm Transformer block: causal MHSA and SwiGLU FFN,
        each wrapped as x + sublayer(RMSNorm(x)).

        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
            rope (bool | None): If truthy, apply RoPE to queries and keys.
            max_seq_len (int): Maximum sequence length for the RoPE cos/sin tables.
            theta (float): RoPE Θ value.
            device (torch.device | None): Device to create parameters and buffers on.
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.attn = MHSA(d_model, num_heads, rope, theta, max_seq_len, device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = FFNN(d_model, d_ff, device, func='SwiGLU')

    def forward(self, x: torch.Tensor):
        x1 = x + self.attn(self.ln1(x))
        result = x1 + self.ffn(self.ln2(x1))
        return result
