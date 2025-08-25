import argparse
import pathlib
import time
import json

from bpe_builder import train_bpe, ENCODING

DEFAULT_DATASET_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "tests/fixtures"
    / "tinystories_sample_5M.txt"
)

parser = argparse.ArgumentParser(description="Train BPE tokenizer on given dataset.")

parser.add_argument("--file", type=str, default=str(DEFAULT_DATASET_PATH))
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
parser.add_argument("--use_cpp", action="store_true")
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--dump_merges", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    start = time.time()
    vocab, merges = train_bpe(args.file, args.vocab_size, args.special_tokens, use_cpp=args.use_cpp)
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
