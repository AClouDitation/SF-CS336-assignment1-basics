import os
import math
import torch
import numpy as np
import random

from typing import Iterable, BinaryIO, IO
from jaxtyping import Int


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it <= cosine_cycle_iters:
        total_cosine_iters = cosine_cycle_iters - warmup_iters
        curr_cosine_iters = it - warmup_iters
        coeff = (math.cos(curr_cosine_iters / total_cosine_iters * math.pi) + 1) / 2
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])
    l2_norm = grads.norm()
    if l2_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad * (max_l2_norm / (l2_norm + 1e-6))


def get_batch(
    x: np.ndarray[tuple[int], np.dtype[np.uint32]],
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
) -> tuple[
    Int[torch.Tensor, "batch_size seq_len"],
    Int[torch.Tensor, "batch_size seq_len"],
]:
    starting_indices = [
        random.randint(0, len(x) - seq_len - 1) for _ in range(batch_size)
    ]
    sequences = torch.LongTensor(np.concatenate(
        [np.array(x[idx : idx + seq_len], dtype=np.int32) for idx in starting_indices]
    )).reshape(batch_size, seq_len).to(device)
    targets = torch.LongTensor(np.concatenate(
        [np.array(x[idx + 1 : idx + seq_len + 1], dtype=np.int32) for idx in starting_indices]
    )).reshape(batch_size, seq_len).to(device)

    return sequences, targets


def save_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration" : iteration,
    }

    torch.save(obj, out)


def load_ckpt(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    assert (
        "model" in obj and "optimizer" in obj and "iteration" in obj
    ), "Checkpoint file is missing required keys."
    assert isinstance(
        obj["iteration"], int
    ), "Iteration in checkpoint is not an integer."

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
