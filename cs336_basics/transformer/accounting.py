import torch
import regex as re
from pprint import pprint

from cs336_basics.transformer import modules


CostBreakdown = modules.CostBreakdown


def total_cost(
    breakdown: CostBreakdown, proj_only: bool = False, is_proj: bool = False
) -> int:
    if isinstance(breakdown, int):
        return breakdown if not proj_only or is_proj else 0
    return sum(total_cost(v, proj_only, "proj" in k) for k, v in breakdown.items())


def aggregate_cost(
    breakdown: CostBreakdown,
    path: str = "",
    proj_only: bool = False,
) -> CostBreakdown:
    if isinstance(breakdown, int):
        raise ValueError("Should never reach here.")
    agg_breakdown = {}
    for k, v in breakdown.items():
        if isinstance(v, int):
            if not proj_only or "proj" in k:
                agg_breakdown[path + k] = v
        else:
            if idx_match := re.search(r"\[\d+\]", k):
                s, e = idx_match.span()
                k = k[:s] + k[e:]
            agg = aggregate_cost(v, f"{path}{k}.", proj_only)
            assert isinstance(agg, dict), f"Expected dict, got {type(agg)}"
            for k, v in agg.items():
                if k not in agg_breakdown:
                    agg_breakdown[k] = 0
                agg_breakdown[k] += v
    return agg_breakdown


if __name__ == "__main__":
    # GPT-2 XL size
    for k, (num_layers, d_model, num_heads) in {
        "GPT-2 XL": (48, 1600, 25),
        "GPT-2 L": (36, 1280, 20),
        "GPT-2 M": (24, 1024, 16),
        "GPT-2 S": (12, 768, 12),
    }.items():
        print(f"Loading model of shape {k}...")
        lm = modules.TransformerLM(
            vocab_size=50257,
            context_length=1024,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=6400,
            rope_theta=0,
        )

        print(f"{k}:")
        print("============================PARAM============================")
        agg_param_cost = aggregate_cost(lm.param_cnt())
        print("FP32 params cost:")
        pprint(
            {k: f"{v / 1e6 * 4:.2f} MB" for k, v in agg_param_cost.items()},  # type: ignore
        )
        print("Total FP32 params cost: %.2f GB\n" % (total_cost(agg_param_cost) / 1e9 * 4))

        print("============================FLOPs============================")
        agg_flops_cost = aggregate_cost(
            lm.forward_flops(tensor_shape=torch.Size((1, 1024, 1600))), proj_only=True
        )
        pprint(
            {k: f"{v / 1e9:.2f} GFLOPs" for k, v in agg_flops_cost.items()},  # type: ignore
        )
        print()
        print(
            "Forward Total FLOPS: %.2f TFLOPs"
            % (
                total_cost(lm.forward_flops(tensor_shape=torch.Size((1, 1024, 1600))))
                / 1e12
            )
        )
        print(
            "Forward Total FLOPS matmul only: %.2f TFLOPs"
            % (
                total_cost(
                    lm.forward_flops(tensor_shape=torch.Size((1, 1024, 1600))),
                    proj_only=True,
                )
                / 1e12
            )
        )
        print("=============================================================")
        print()
        del lm
