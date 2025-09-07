import math


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
