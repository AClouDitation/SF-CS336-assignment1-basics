import os
import argparse
import pathlib
import time
import json

from cs336_basics import bpe_tokenization
from cs336_basics.bpe_tokenization.bpe_builder import ENCODING, BytePairEncodingBuilder
from cs336_basics.bpe_tokenization.pretokenization import pretokenize

DEFAULT_DATASET_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "tests/fixtures"
    / "tinystories_sample_5M.txt"
)

parser = argparse.ArgumentParser(description="Train BPE tokenizer on given dataset.")

parser.add_argument("--file", type=str, default=str(DEFAULT_DATASET_PATH))
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--dump_merges", action="store_true")

parser.add_argument("--use_cpp", action="store_true")
parser.add_argument("--recompile", action="store_true")


def train_bpe(
    file_path: str | os.PathLike,
    target_vocab_size: int,
    special_tokens: list[str],
    use_cpp: bool = False,
    recompile: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if use_cpp:
        if recompile:
            bpe_tokenization.compile_cc()
        bpe_tokenization.load_cpp_libs()

        from cppyy.gbl import bpe as cc  # type: ignore
        cc.BPEBuilder.Train.__release_gil__ = True

        builder = cc.BPEBuilder(
            special_tokens=special_tokens,
            target_vocab_size=target_vocab_size,
        )

        for pretoken, count in pretokenize(
            file_path, [t.encode(ENCODING) for t in special_tokens]
        ).items():
            builder.AddPretoken(pretoken, count)

        builder.Train()
        vocab = builder.GetVocab()
        merges = builder.GetMerges()

        return {i: bytes(t) for i, t in enumerate(vocab)}, [
            (merge.first, merge.second) for merge in merges
        ]  # type:ignore

    builder = BytePairEncodingBuilder(file_path, target_vocab_size, special_tokens)
    builder.train()

    return builder.vocab, builder.merges


if __name__ == "__main__":
    args = parser.parse_args()

    start = time.time()
    vocab, merges = train_bpe(
        args.file,
        args.vocab_size,
        args.special_tokens,
        use_cpp=args.use_cpp,
        recompile=args.recompile,
    )
    dur = time.time() - start
    print(f"Training finished. Took: {dur:.2f} seconds.")

    with open(pathlib.Path(args.output_dir) / "vocab.json", "w") as f:
        json.dump({k: v.decode(ENCODING, errors="ignore") for k, v in vocab.items()}, f)

    if args.dump_merges:
        with open(pathlib.Path(args.output_dir) / "merges.txt", "w") as f:
            f.writelines(
                [
                    f"{merge[0].decode(ENCODING, errors="ignore")} {merge[1].decode(ENCODING, errors="ignore")}\n"
                    for merge in merges
                ]
            )
