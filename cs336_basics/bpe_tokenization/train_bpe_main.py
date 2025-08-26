import os
import argparse
import pathlib
import time
import json

from datasets import load_dataset
from typing import Callable
from cs336_basics import bpe_tokenization
from cs336_basics.bpe_tokenization import pretokenization

ENCODING = "utf-8"

DEFAULT_DATASET_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "tests/fixtures"
    / "tinystories_sample_5M.txt"
)

parser = argparse.ArgumentParser(description="Train BPE tokenizer on given dataset.")

parser.add_argument("--file", type=str, default=str(DEFAULT_DATASET_PATH))
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--output_name", type=str, default="vocab.json")
parser.add_argument("--output_merges_name", type=str, default=None)
parser.add_argument("--recompile", action="store_true")


def train_bpe(
    pretokenize_fn: Callable[[], dict[bytes, int]],
    target_vocab_size: int,
    special_tokens: list[str],
    recompile: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if recompile:
        bpe_tokenization.compile_cc()
    bpe_tokenization.load_cpp_libs()

    from cppyy.gbl import bpe as cc  # type: ignore

    cc.BPEBuilder.Train.__release_gil__ = True

    builder = cc.BPEBuilder(
        special_tokens=special_tokens,
        target_vocab_size=target_vocab_size,
    )

    for pretoken, count in pretokenize_fn().items():
        builder.AddPretoken(pretoken, count)

    builder.Train()
    vocab = builder.GetVocab()
    merges = builder.GetMerges()

    return {i: bytes(t) for i, t in enumerate(vocab)}, [
        (merge.first, merge.second) for merge in merges
    ]  # type:ignore


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dataset:
        match args.dataset:
            case "owt":
                pretokenize_fn = pretokenization.pretokenize_owt
            case _:
                raise ValueError(f"Unknown dataset: {args.dataset}")
    elif args.file:
        pretokenize_fn = lambda: pretokenization.pretokenize_from_file(
            args.file, [t.encode(ENCODING) for t in args.special_tokens]
        )
    else:
        raise ValueError("Either --file or --dataset must be provided.")

    start = time.time()
    vocab, merges = train_bpe(
        pretokenize_fn=pretokenize_fn,
        target_vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        recompile=args.recompile,
    )
    dur = time.time() - start
    print(f"Training finished. Took: {dur:.2f} seconds.")

    with open(pathlib.Path(args.output_dir) / args.output_name, "w") as f:
        json.dump({k: v.decode(ENCODING, errors="ignore") for k, v in vocab.items()}, f)

    if args.output_merges_name:
        with open(pathlib.Path(args.output_dir) / args.output_merges_name, "w") as f:
            f.writelines(
                [
                    f"{merge[0].decode(ENCODING, errors="ignore")} {merge[1].decode(ENCODING, errors="ignore")}\n"
                    for merge in merges
                ]
            )
