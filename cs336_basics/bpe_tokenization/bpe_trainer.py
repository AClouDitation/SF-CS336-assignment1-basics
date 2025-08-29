from typing import Callable
from cs336_basics.bpe_tokenization import cc

def train_bpe(
    pretokenize_fn: Callable[[], dict[bytes, int]],
    target_vocab_size: int,
    special_tokens: list[str],
    recompile: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if recompile:
        cc.compile()
        cc.load_libs()

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
        (bytes(merge.first), bytes(merge.second)) for merge in merges
    ]  # type:ignore
