import time
import os
import pathlib
import cProfile
import pstats

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from cs336_basics.bpe_tokenization.tokenizer import Tokenizer
from cs336_basics.bpe_tokenization.pretokenization import PAT


if __name__ == "__main__":
    from tests.common import FIXTURES_PATH
    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

    num_processes = os.cpu_count() or 1
    dataset: Dataset = load_dataset("roneneldan/TinyStories", num_proc=num_processes, split="train")  # type: ignore
    dataset = dataset.shuffle(seed=int(time.time()))

    examples = dataset[:1000]["text"]
    total_bytes = sum(len(e) for e in examples)
    print(f"Total bytes to encode: {total_bytes} bytes")

    # dir = pathlib.Path(__file__).resolve().parent
    tokenizer: Tokenizer = get_tokenizer_from_vocab_merges_path(
        # vocab_file=dir / "tiny_stories/vocab_10k.json",
        # merges_file=dir / "tiny_stories/merges_10k.txt",
        FIXTURES_PATH / "gpt2_vocab.json",
        FIXTURES_PATH / "gpt2_merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    print(f"Tokenizer vocab size {len(tokenizer._vocab)}")
    profiler = cProfile.Profile()
    start = time.time()
    profiler.enable()
    token_ids = list(tokenizer.encode_iterable(examples))
    profiler.disable()
    end = time.time()

    # print(f"Number of tokens: {len(token_ids)}")
    # print(f"Compression ratio: {total_bytes / len(token_ids)}")
    print(f"Duration: {end - start:.2f} seconds")
    # print(f"Throughput: {total_bytes / (end - start):.2f} bytes/second")

    import tiktoken
    reference_tokenizer = tiktoken.get_encoding("gpt2")

    ref_profiler = cProfile.Profile()
    start = time.time()
    ref_profiler.enable()
    ref_token_ids = []
    for example in examples:
        ref_token_ids.extend(reference_tokenizer.encode(example))
    ref_profiler.disable()
    end = time.time()

    # print(f"Number of tokens: {len(token_ids)}")
    # print(f"Compression ratio: {total_bytes / len(token_ids)}")
    print(f"Ref Duration: {end - start:.2f} seconds")
    # print(f"Throughput: {total_bytes / (end - start):.2f} bytes/second")

    print("Tokenizer Profiling Stats:")
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME).print_stats(20)

    print("Ref Tokenizer Profiling Stats:")
    stats = pstats.Stats(ref_profiler).sort_stats(pstats.SortKey.TIME).print_stats(20)