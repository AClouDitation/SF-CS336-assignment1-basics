import os
import glob
import numpy as np

from typing import Iterable


def get_path_for_shard(final_path: str | os.PathLike, shard_index: int) -> str:
    path, basename = os.path.split(final_path)
    fn, ext = os.path.splitext(basename)
    return os.path.join(path, f"{fn}_part_{shard_index}{ext}")

def _get_shards(final_path: str | os.PathLike) -> Iterable[str]:
    path, basename = os.path.split(final_path)
    fn, ext = os.path.splitext(basename)
    return glob.glob(os.path.join(path, f"{fn}_part_*{ext}"))


def _cleanup_shard_files(final_path: str | os.PathLike):
    for f in _get_shards(final_path):
        os.remove(f)


def merge_memmaps(
    output_path: str | os.PathLike, dtype=np.uint32
) -> np.memmap:
    shards_path = _get_shards(output_path)
    total_size = sum(os.path.getsize(f) // np.dtype(dtype).itemsize for f in shards_path)

    output = np.memmap(output_path, dtype=dtype, mode='w+', shape=(total_size,))
    pos = 0
    for file_path in shards_path:
        data = np.memmap(file_path, dtype=dtype, mode='r')
        output[pos:pos+len(data)] = data
        pos += len(data)

    output.flush()

    _cleanup_shard_files(output_path)
    return np.memmap(output_path, dtype=dtype, mode='r')