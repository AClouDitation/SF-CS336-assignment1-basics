import einx
import torch

from torch import Tensor
from jaxtyping import Float, Bool


def soft_max(x: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
    x = x - x.max(dim=dim, keepdim=True).values
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[Tensor, "batch_size ... n d_k"],
    k: Float[Tensor, "batch_size ... m d_k"],
    v: Float[Tensor, "batch_size ... m d_v"],
    mask: Bool[Tensor, "batch_size ... n m"] | None = None,
) -> Float[Tensor, "batch_size ... d_v"]:
    pre_softmax = einx.dot("... n d_k, ... m d_k -> ... n m", q, k)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, -torch.inf)

    t = pre_softmax / (k.shape[-1] ** 0.5)
    return einx.dot("... n m, ... m d_v -> ... n d_v", soft_max(t), v)
