import time
import os
import pathlib

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from cs336_basics.bpe_tokenization.tokenizer import Tokenizer


if __name__ == "__main__":

    dir = pathlib.Path(__file__).resolve().parent
    tokenizer = Tokenizer.from_files(
        vocab_file=dir / "tiny_stories/vocab_10k.json",
        merges_file=dir / "tiny_stories/merges_10k.txt",
    )

    num_processes = os.cpu_count() or 1

    dataset: Dataset = load_dataset("roneneldan/TinyStories", num_proc=num_processes, split="train")  # type: ignore
    dataset.shuffle(seed=int(time.time()))

    examples = dataset[:1000]["text"]
    total_bytes = sum(len(e) for e in examples)
    print(f"Total bytes to encode: {total_bytes} bytes")

    start = time.time()
    token_ids = list(tokenizer.encode_iterable(examples))
    end = time.time()

    print(f"Number of tokens: {len(token_ids)}")
    print(f"Compression ratio: {total_bytes / len(token_ids)}")
    print(f"Duration: {end - start:.2f} seconds")
    print(f"Throughput: {total_bytes / (end - start):.2f} bytes/second")

    import tiktoken
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    # reference_tokenizer = tiktoken._educational.SimpleBytePairEncoding.from_tiktoken("gpt2")

    start = time.time()
    token_ids = list(
        reference_tokenizer.encode(
            "<|endoftext|>".join(examples), allowed_special={"<|endoftext|>"}
        )
    )
    end = time.time()

    print(f"Number of tokens: {len(token_ids)}")
    print(f"Compression ratio: {total_bytes / len(token_ids)}")
    print(f"Duration: {end - start:.2f} seconds")
    print(f"Throughput: {total_bytes / (end - start):.2f} bytes/second")
