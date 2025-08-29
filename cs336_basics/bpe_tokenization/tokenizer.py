import json
import os
import pathlib

from base64 import b64decode
from cs336_basics.bpe_tokenization import pretokenization, ENCODING
from typing import Iterable, Iterator
from collections import defaultdict


class Node:
    def __init__(self, token: bytes):
        self.token = token
        self.next: Node | None = None


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

        self._merges = merges
        self._pretoken_cache: dict[bytes, list[int]] = {}

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
                        b64decode(merged_tokens[0].encode(ENCODING)),
                        b64decode(merged_tokens[1].encode(ENCODING)),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        if not pretoken:
            return []
        if pretoken in self._pretoken_cache:
            return self._pretoken_cache[pretoken]

        existing_token = defaultdict(list[Node])
        head = Node(pretoken[0].to_bytes())

        existing_token[pretoken[0].to_bytes()].append(head)
        tail = head

        for b in pretoken[1:]:
            tail.next = Node(b.to_bytes())
            tail = tail.next
            existing_token[b.to_bytes()].append(tail)

        for merge in self._merges:
            if merge[0] not in existing_token:
                continue

            curr = head
            for node in existing_token[merge[0]]:
                if node.next and (node.token, node.next.token) == merge:
                    node.token = merge[0] + merge[1]
                    node.next = node.next.next
                    existing_token[merge[0] + merge[1]].append(node)

        token_ids: list[int] = []
        curr = head
        while curr:
            token_ids.append(self._reverse_vocab[curr.token])
            curr = curr.next
        self._pretoken_cache[pretoken] = token_ids
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
    # print("TokenIds:", token_ids)
    print(f"Encode Duration: {time.time() - start:.2f}")

    start = time.time()
    decoded_text = t.decode(token_ids)
    # print("Decoded Text:", decoded_text)
    print(f"Decode Duration: {time.time() - start:.2f}")
