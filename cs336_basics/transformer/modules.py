import torch
import einx

from torch import Tensor
from jaxtyping import Float
from cs336_basics.transformer import utils


class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.W: Float[Tensor, "d_out d_in"] = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einx.dot("... d_in, d_out d_in -> ... d_out", x, self.W)


class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding: Float[Tensor, "d_vocab d_model"] = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.embedding, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._d_model = d_model
        self._eps = eps
        self._dtype = dtype
        self.gain: Float[Tensor, "d_model"] = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x = x.to(torch.float32)
        rms = (einx.sum("... d_model -> ...", x**2) / self._d_model + self._eps) ** 0.5
        x = einx.divide("... d_model, ... -> ... d_model", x, rms)
        x = einx.multiply("... d_model, d_model -> ... d_model", x, self.gain)
        return x.to(self._dtype)


class SwiGLU(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W1: Float[Tensor, "d_ff d_model"] = torch.nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.W2: Float[Tensor, "d_model d_ff"] = torch.nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.W3: Float[Tensor, "d_ff d_model"] = torch.nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        def silu(x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)

        W1x = einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.W1, x)
        W3x = einx.dot("d_ff d_model, ... d_model -> ... d_ff", self.W3, x)
        t = einx.multiply("... d_ff, ... d_ff -> ... d_ff", silu(W1x), W3x)
        return einx.dot("d_model d_ff, ... d_ff -> ... d_model", self.W2, t)


class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        super().__init__()
        seq = torch.arange(0, max_seq_len, device=device)
        k = torch.arange(0, d_k, 2, device=device)
        theta_mat = einx.dot(
            "seq, half_d_k -> seq half_d_k", seq, 1 / theta ** (k / d_k)
        )
        cos = theta_mat.cos().repeat_interleave(2, dim=-1)
        sin = theta_mat.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Float[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        swapped_x = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        return x * self.cos[token_positions] + swapped_x * self.sin[token_positions]  # type: ignore

class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope_module: RotaryPositionalEmbedding | None = None,
    ):
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads, d_model=%d, num_heads=%d" % (
            d_model,
            num_heads,
        )
        super().__init__()

        self.num_heads = num_heads
        self._rope_module = rope_module

        self.Wq: Float[Tensor, "d_k d_model"] = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )
        self.Wk: Float[Tensor, "d_k d_model"] = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )
        self.Wv: Float[Tensor, "d_v d_model"] = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )
        self.Wo: Float[Tensor, "d_model d_v"] = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        token_positions: Float[Tensor, "... seq"] | None = None,
    ) -> Float[Tensor, "... d_model"]:
        Wq: Tensor = einx.rearrange(
            "(h hd_k) d_model -> h hd_k d_model", self.Wq, h=self.num_heads
        )  # type:ignore
        Wk: Tensor = einx.rearrange(
            "(h hd_k) d_model -> h hd_k d_model", self.Wk, h=self.num_heads
        )  # type:ignore
        Wv: Tensor = einx.rearrange(
            "(h hd_v) d_model -> h hd_v d_model", self.Wv, h=self.num_heads
        )  # type:ignore

        Q = einx.dot("h hd_k d_model, ... seq d_model -> ... h seq hd_k", Wq, x)
        K = einx.dot("h hd_k d_model, ... seq d_model -> ... h seq hd_k", Wk, x)
        if self._rope_module is not None and token_positions is not None:
            Q = self._rope_module(Q, token_positions)
            K = self._rope_module(K, token_positions)

        V = einx.dot("h hd_v d_model, ... seq d_model -> ... h seq hd_v", Wv, x)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).unsqueeze(0)
        mask = mask.to(Q.device)

        attn: Tensor = einx.rearrange(
            "... h seq hd_k -> ... seq (h hd_k)",
            utils.scaled_dot_product_attention(Q, K, V, mask),
        )  # type:ignore

        return einx.dot("d_v d_model, ... seq d_model -> ... seq d_v", self.Wo, attn)
