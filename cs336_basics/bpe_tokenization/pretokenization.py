import os
import regex as re

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

PAT = re.compile(
    rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    regex_pattern: re.Pattern | bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = re.search(regex_pattern, mini_chunk)
            if found_at:
                chunk_boundaries[bi] = initial_position + found_at.start()
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _read_and_split_chunk(
    file_path: str | os.PathLike, start: int, end: int, regex_pattern: re.Pattern
) -> list[bytes]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    return [m for m in re.split(regex_pattern, chunk)]


def _count_pretoken(chunk: bytes) -> dict[bytes, int]:
    pretoken_cnt: dict[bytes, int] = defaultdict(int)
    for pretoken in re.findall(PAT, chunk):
        pretoken_cnt[pretoken] += 1
    return pretoken_cnt


def _read_and_count_pretoken(
    file_path: str | os.PathLike, start: int, end: int, regex_pattern: re.Pattern
) -> dict[bytes, int]:
    pretoken_cnt: dict[bytes, int] = defaultdict(int)

    for chunk in _read_and_split_chunk(file_path, start, end, regex_pattern):
        for k, v in _count_pretoken(chunk).items():
            pretoken_cnt[k] += v

    return pretoken_cnt


def pretokenize_from_file(
    file_path: str | os.PathLike, separators: list[bytes]
) -> dict[bytes, int]:
    regex_pattern = re.compile(b"|".join(map(re.escape, separators)))
    num_processes = os.cpu_count() or 1

    with open(file_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processes, regex_pattern)

    print("Loading %d chunks" % (len(boundaries) - 1))
    if len(boundaries) > 2:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                futures.append(
                    executor.submit(
                        _read_and_count_pretoken, file_path, start, end, regex_pattern
                    )
                )

        pretoken_cnt: dict[bytes, int] = defaultdict(int)
        for future in futures:
            for k, v in future.result().items():
                pretoken_cnt[k] += v
    else:
        pretoken_cnt = _read_and_count_pretoken(
            file_path, boundaries[0], boundaries[1], regex_pattern
        )

    return pretoken_cnt


def count_owt_example(dataset: Dataset):
    pretoken_cnt: dict[bytes, int] = defaultdict(int)
    for example in dataset:
        for k, v in _count_pretoken(example["text"].encode("utf-8")).items():  # type: ignore
            pretoken_cnt[k] += v
    return pretoken_cnt


def pretokenize_owt() -> dict[bytes, int]:
    num_processes = os.cpu_count() or 1

    dataset: Dataset = load_dataset("Skylion007/openwebtext", num_proc=num_processes, split="train")  # type: ignore
    total_chunks = len(dataset)
    print("Loading %d chunks..." % total_chunks, end='')
    shards = [
        dataset.shard(num_shards=num_processes, index=i) for i in range(num_processes)
    ]
    print("into %d shards" % len(shards))
    print("PID: %d" % os.getpid())
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for shard in shards:
            futures.append(executor.submit(count_owt_example, shard))
    print("Done")

    pretoken_cnt: dict[bytes, int] = defaultdict(int)
    for future in futures:
        for k, v in future.result().items():
            pretoken_cnt[k] += v
    return pretoken_cnt
