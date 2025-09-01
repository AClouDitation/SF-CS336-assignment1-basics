from torch import Tensor
from jaxtyping import Float


def soft_max(x: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
    x = x - x.max(dim=dim, keepdim=True).values
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)