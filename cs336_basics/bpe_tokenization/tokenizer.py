import itertools
import json
import os
import pathlib

from base64 import b64decode
from cs336_basics.bpe_tokenization import pretokenization, ENCODING
from typing import Iterable, Iterator


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._reverse_vocab = {v: k for k, v in vocab.items()}
        self._special_tokens = (
            sorted(
                [token.encode(ENCODING) for token in special_tokens],
                key=len,
                reverse=True,
            )
            if special_tokens
            else []
        )

        for special_token in self._special_tokens:
            if special_token not in self._reverse_vocab:
                self._reverse_vocab[special_token] = len(self._reverse_vocab)

        self._vocab = [b""] * len(self._reverse_vocab)
        for token, i in self._reverse_vocab.items():
            self._vocab[i] = token

        # self._len_sorted_vocab = sorted(self.vocab, key=len, reverse=True)
        self._merges = merges
        self._cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_file: str | os.PathLike,
        merges_file: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_file, "r", encoding=ENCODING) as f:
            vocab = {v: b64decode(k.encode(ENCODING)) for k, v in json.load(f).items()}
        with open(merges_file, "r", encoding=ENCODING) as f:
            merges = []
            for line in f.readlines():
                merged_tokens = line.split(" ")
                assert len(merged_tokens) == 2, f"Invalid merge line: {line}"
                merges.append(
                    (
                        merged_tokens[0].encode(ENCODING),
                        merged_tokens[1].encode(ENCODING),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        if pretoken in self._cache:
            return self._cache[pretoken]
        tokens: list[bytes] = [b.to_bytes() for b in pretoken]
        for merge in self._merges:
            to_replace: list[int] = []
            skip_next = False
            for idx, pair in enumerate(itertools.pairwise(tokens)):
                if skip_next:
                    skip_next = False
                    continue
                if pair == merge:
                    to_replace.append(idx)
                    skip_next = True

            for idx in to_replace[::-1]:
                tokens[idx] = merge[0] + merge[1]
                tokens.pop(idx + 1)

        token_ids = [self._reverse_vocab[token] for token in tokens]
        self._cache[pretoken] = token_ids
        return token_ids

    def encode(self, text: str) -> list[int]:
        pretokens = pretokenization.pretokenize(
            text.encode(ENCODING), separators=self._special_tokens
        )

        pretoken_token_ids: dict[bytes, list[int]] = {}
        for pretoken in set(pretokens):
            if pretoken in self._special_tokens:
                pretoken_token_ids[pretoken] = [self._reverse_vocab[pretoken]]
            else:
                pretoken_token_ids[pretoken] = self._encode_pretoken(pretoken)

        return [id for pretoken in pretokens for id in pretoken_token_ids[pretoken]]

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self._vocab[i] for i in token_ids]
        return b''.join(tokens).decode(ENCODING, errors="replace")

if __name__ == "__main__":
    from tests.common import FIXTURES_PATH
    import time

    t = Tokenizer.from_files(
        pathlib.Path(__file__).resolve().parent / "tiny_stories_5m/vocab.json",
        pathlib.Path(__file__).resolve().parent / "tiny_stories_5m/merges.txt",
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )

    with open(FIXTURES_PATH / "tinystories_sample_5M.txt", "r", encoding=ENCODING) as f:
        text = f.read()
    start = time.time()
    token_ids = t.encode(text)
    print("TokenIds:", token_ids)
    decoded_text = t.decode(token_ids)
    print(f"Duration: {time.time() - start:.2f}")
    print("Decoded Text:", decoded_text)
