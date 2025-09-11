import json
import os
import heapq
import numpy as np

from base64 import b64decode
from cs336_basics.bpe_tokenization import pretokenization, ENCODING
from typing import Iterable, Iterator, Any
from cs336_basics.common import memmap_utils


class Node:
    def __init__(self, token: bytes):
        self.token = token
        self.prev: Node | None = None
        self.next: Node | None = None

    def __repr__(self) -> str:
        curr = self
        tokens = []
        while curr:
            tokens.append(curr.token)
            curr = curr.next
        return "<->".join([f"'{t.decode(ENCODING, errors="replace")}'" for t in tokens])
    
    def __lt__(self, other: "Node") -> bool:
        return id(self) < id(other)


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._reverse_vocab = {v: k for k, v in vocab.items()}
        self._special_tokens = set()
        if special_tokens: 
            self._special_tokens = set(token.encode(ENCODING) for token in special_tokens)
            for special_token in sorted(self._special_tokens, key=len, reverse=True):
                if special_token not in self._reverse_vocab:
                    self._reverse_vocab[special_token] = len(self._reverse_vocab)

        self._vocab = [b""] * len(self._reverse_vocab)
        for token, i in self._reverse_vocab.items():
            self._vocab[i] = token

        self._merges = merges
        self._pretoken_cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def special_tokens(self) -> list[bytes]:
        return list(self._special_tokens)

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

        head = Node(pretoken[0].to_bytes())
        curr = head
        for b in pretoken[1:]:
            curr.next = Node(b.to_bytes())
            curr = curr.next

        while True:
            min_node = None
            min_rank = None
            curr = head
            while curr and curr.next:
                rank = self._reverse_vocab.get(curr.token + curr.next.token)
                if rank is not None:
                    if min_rank is None or rank < min_rank:
                        min_rank = rank
                        min_node = curr
                curr = curr.next

            if min_rank is None:
                break
            # assert min_node and min_node.next

            min_node.token = min_node.token + min_node.next.token  # type: ignore
            min_node.next = min_node.next.next  # type: ignore

        token_ids: list[int] = []
        curr = head
        while curr:
            token_ids.append(self._reverse_vocab[curr.token])
            curr = curr.next

        self._pretoken_cache[pretoken] = token_ids
        return token_ids

    def _encode_pretoken_v2(self, pretoken: bytes) -> list[int]:
        if not pretoken:
            return []
        if pretoken in self._pretoken_cache:
            return self._pretoken_cache[pretoken]

        head = Node(pretoken[0].to_bytes())
        curr = head
        heap = []
        for b in pretoken[1:]:
            new_node = Node(b.to_bytes())
            curr.next = new_node
            new_node.prev = curr

            if rank := self._reverse_vocab.get(curr.token + new_node.token):
                heap.append((rank, curr))

            curr = new_node
        heapq.heapify(heap)

        while heap:
            min_rank, min_node = heapq.heappop(heap)
            if not min_node.next:
                continue
            pair = min_node.token + min_node.next.token
            if min_rank != self._reverse_vocab.get(pair):
                continue  # Outdated heap entry

            min_node.token = pair
            to_remove = min_node.next
            min_node.next = to_remove.next
            if to_remove.next:
                to_remove.next.prev = min_node
                to_remove.next = None
                to_remove.prev = None

            if min_node.prev:
                pair = min_node.prev.token + min_node.token
                rank = self._reverse_vocab.get(pair)
                if rank is not None:
                    heapq.heappush(heap, (rank, min_node.prev))

            if min_node.next:
                pair = min_node.token + min_node.next.token
                rank = self._reverse_vocab.get(pair)
                if rank is not None:
                    heapq.heappush(heap, (rank, min_node))

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
                pretoken_token_ids[pretoken] = self._encode_pretoken_v2(pretoken)

        return [id for pretoken in pretokens for id in pretoken_token_ids[pretoken]]

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        tokens = [self._vocab[i] for i in token_ids]
        return b"".join(tokens).decode(ENCODING, errors="replace")

    def encode_file(
        self,
        input_path: str | os.PathLike,
        output_path: str | os.PathLike,
    ) -> np.memmap[tuple[int], np.dtype[np.uint32]]:
        num_processes = os.cpu_count() or 1
        with open(input_path, "rb") as f_in:
            boundaries = pretokenization._find_chunk_boundaries(
                f_in, num_processes, list(self._special_tokens)
            )

        def get_file_chunk() -> Iterator[str]:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                with open(input_path, "r") as f_in:
                    f_in.seek(start)
                    yield f_in.read(end - start)

        for i, chunk in enumerate(get_file_chunk()):
            np.save(
                memmap_utils.get_path_for_shard(output_path, i),
                np.array(self.encode(chunk), dtype=np.uint32),
            )
        return memmap_utils.merge_memmaps(output_path, dtype=np.uint32)


if __name__ == "__main__":
    from tests.common import FIXTURES_PATH
    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    import time

    # t = Tokenizer.from_files(
    #     vocab_file=dir / "tiny_stories/vocab_10k.json",
    #     merges_file=dir / "tiny_stories/merges_10k.txt",
    #     special_tokens=[],
    # )
    t = get_tokenizer_from_vocab_merges_path(
        FIXTURES_PATH / "gpt2_vocab.json",
        FIXTURES_PATH / "gpt2_merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    # with open(FIXTURES_PATH / "tinystories_sample_5M.txt", "r", encoding=ENCODING) as f:
    #     text = f.read()
    start = time.time()
    token_ids = t.encode("Lily")
    print("TokenIds:", token_ids)
    print("Tokens:", [t.decode([id]) for id in token_ids])
    print(f"Encode Duration: {time.time() - start:.2f}")

    start = time.time()
    decoded_text = t.decode(token_ids)
    print("Decoded Text:", decoded_text)
    print(f"Decode Duration: {time.time() - start:.2f}")
