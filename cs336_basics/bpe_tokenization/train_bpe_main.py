import argparse
import pathlib
import time
import json

from base64 import b64encode
from cs336_basics.bpe_tokenization import pretokenization, bpe_trainer, ENCODING


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
parser.add_argument("--output_vocab_name", type=str, default="vocab.json")
parser.add_argument("--output_merges_name", type=str, default="merges.txt")
parser.add_argument("--recompile", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dataset:
        pretokenize_fn = lambda: pretokenization.count_pretokens_from_hugging_face(
            args.dataset
        )
    elif args.file:
        pretokenize_fn = lambda: pretokenization.count_pretokens_from_file(
            args.file, [t.encode(ENCODING) for t in args.special_tokens]
        )
    else:
        raise ValueError("Either --file or --dataset must be provided.")

    start = time.time()
    vocab, merges = bpe_trainer.train_bpe(
        pretokenize_fn=pretokenize_fn,
        target_vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        recompile=args.recompile,
    )
    dur = time.time() - start
    print(f"Training finished. Took: {dur:.2f} seconds.")

    with open(pathlib.Path(args.output_dir) / args.output_vocab_name, "w") as f:
        json.dump(
            {b64encode(v).decode(ENCODING): k for k, v in vocab.items()},
            f,
        )

    with open(pathlib.Path(args.output_dir) / args.output_merges_name, "w") as f:
        f.writelines(
            [
                f"{b64encode(merge[0]).decode(ENCODING)} {b64encode(merge[1]).decode(ENCODING)}\n"
                for merge in merges
            ]
        )
