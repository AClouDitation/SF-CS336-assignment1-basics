import torch

from typing import Callable


class AdamW(torch.optim.Optimizer):

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ): 
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None):  # type: ignore
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                m = state.get("m", torch.zeros_like(p.data))
                m = b1 * m + (1 - b1) * p.grad.data
                state["m"] = m

                v = state.get("v", torch.zeros_like(p.data))
                v = b2 * v + (1 - b2) * p.grad.data**2
                state["v"] = v

                t = state.get("t", 1)
                lr_t = lr * (1 - b2**t) ** 0.5 / (1 - b1**t)
                p.data -= lr_t * m / (v.sqrt() + eps) 
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1

        return loss
