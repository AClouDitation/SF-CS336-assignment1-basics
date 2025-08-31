import torch
import einx

from torch import Tensor
from jaxtyping import Float


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
