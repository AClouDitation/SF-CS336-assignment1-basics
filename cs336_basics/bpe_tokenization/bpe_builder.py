import logging
import regex as re

from collections import defaultdict, OrderedDict, deque
from typing import TypeAlias
from concurrent.futures import ProcessPoolExecutor
import functools


ENCODING = "utf-8"
ENALBE_MULTI_PROCESSING = False
logger = logging.getLogger(__name__)

BytesPair: TypeAlias = tuple[bytes, bytes]


class Node:
    def __init__(self, val: bytes):
        self.val = val
        self.prev: Node | None = None
        self.next: Node | None = None

    def __repr__(self) -> str:
        nodes = [self]
        curr = self.next
        while curr:
            nodes.append(curr)  # type: ignore
            curr = curr.next

        return " <-> ".join(map(lambda x: f"{repr(x.val)}", nodes))

    @classmethod
    def from_list(cls, lst: list[bytes]) -> "Node":
        assert lst, "List must not be empty"
        head = cls(lst[0])
        current = head
        for item in lst[1:]:
            new_node = cls(item)
            current.next = new_node
            new_node.prev = current
            current = new_node

        return head


class PreToken:
    def __init__(self, value: bytes):
        self.value = value
        self.tokens = Node.from_list([b.to_bytes() for b in value])

        # Guarantees in order of appearance
        self._pairs_heads: dict[BytesPair, OrderedDict[Node, None]] = defaultdict(
            OrderedDict[Node, None]
        )

        curr = self.tokens
        while curr and curr.next:
            pair = (curr.val, curr.next.val)
            self._pairs_heads[pair][curr] = None
            curr = curr.next

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PreToken):
            return self.value == other.value
        raise NotImplementedError()

    @property
    def pairs_cnt(self) -> dict[BytesPair, int]:
        return {k: len(v) for k, v in self._pairs_heads.items()}

    def merge_bytes_pair(self, pair: BytesPair) -> dict[BytesPair, int]:
        """Merges bytes pair in PreToken. Updates pairs_head affected.

        Returns: A dict of diff in pairs count.
        """
        if pair not in self._pairs_heads:
            # logger.info("Pair %s not found in PreToken %s. Skipping..", pair, self)
            return {}
        # logger.info("Merging pair: %s in PreToken %s", pair, self)
        new_token = pair[0] + pair[1]
        new_nodes: dict[Node, Node] = OrderedDict()  # node -> node pair removed
        node_to_skip: set[Node] = set()
        for head in self._pairs_heads[pair]:
            # logger.debug(f"Processing head %#X: %s", id(head), head.val)
            if head in node_to_skip:
                continue

            new_node = Node(new_token)

            # update lhs
            if prev_node := head.prev:
                prev_node.next = new_node
            else:  # updating head
                self.tokens = new_node
            new_node.prev = prev_node

            # update rhs
            assert head.next
            if next_node := head.next.next:
                next_node.prev = new_node
            new_node.next = next_node

            new_nodes[new_node] = head
            head.prev = None
            node_to_skip.add(head.next)

        # logger.debug("new_nodes: ")
        # logger.debug([f"{id(v):#X}: {v.val}" for v in new_nodes])
        # logger.debug("PreToken: %s", self)

        updated_nodes = set()
        diff = defaultdict(int)
        for node, orig_nodes in new_nodes.items():
            if node.prev and node.prev not in updated_nodes:
                new_pair = (node.prev.val, node.val)
                # logger.debug("Adding new Pair %s from previous node", pair)
                self._pairs_heads[new_pair][node.prev] = None
                diff[new_pair] += 1
                # logger.debug("PreToken: %s", self)

                prev_pair = (node.prev.val, pair[0])
                # logger.debug("Updating affected pair %s", prev_pair)
                del self._pairs_heads[prev_pair][node.prev]
                diff[prev_pair] -= 1
                if not self._pairs_heads[prev_pair]:
                    del self._pairs_heads[prev_pair]
                # logger.debug("PreToken: %s", self)

            if node.next:
                new_pair = (node.val, node.next.val)
                # logger.debug("Adding new Pair %s from next node", new_pair)
                self._pairs_heads[new_pair][node] = None
                diff[new_pair] += 1
                # logger.debug("PreToken: %s", self)

            assert orig_nodes.next
            if orig_next := orig_nodes.next.next:
                next_pair = (pair[1], orig_next.val)
                # logger.debug("Updating affected pair %s", next_pair)
                del self._pairs_heads[next_pair][orig_nodes.next]
                diff[next_pair] -= 1
                if not self._pairs_heads[next_pair]:
                    del self._pairs_heads[next_pair]
                # logger.debug("PreToken: %s", self)

            updated_nodes.add(node)

        del self._pairs_heads[pair]

        # logger.debug("Updated: %s", self)
        return diff

    def __repr__(self) -> str:
        # heads_repr = {
        # k: [f"{id(v):#X}: {v.val}" for v in vl]
        # for k, vl in self._pairs_heads.items()
        # }
        heads_repr = {
            k: [f"{v.val}" for v in vl] for k, vl in self._pairs_heads.items()
        }
        return (
            f"PreToken(value={self.value}, tokens={self.tokens})\n"
            # f"pairs_cnt={self.pairs_cnt}\n"
            f"pairs_heads={heads_repr}\n"
        )


class PreTokenV2:
    def __init__(self, value: bytes):
        self.value = value
        self.tokens = [b.to_bytes() for b in value]

        # self._pairs_cnt: dict[BytesPair, int] = defaultdict(int)
        # for idx in range(len(self.tokens) - 1):
        #     pair = self.tokens[idx], self.tokens[idx + 1]

    @property
    def pairs_cnt(self) -> dict[BytesPair, int]:
        # return self._pairs_cnt
        pairs_cnt: dict[BytesPair, int] = defaultdict(int)
        for idx in range(len(self.tokens) - 1):
            pair = self.tokens[idx], self.tokens[idx + 1]
            pairs_cnt[pair] += 1
        return pairs_cnt

    def merge_bytes_pair(self, target_pair: BytesPair) -> dict[BytesPair, int]:
        # if target_pair not in self._pair_heads:
        #     return {}

        new_token = target_pair[0] + target_pair[1]
        last_match_idx = -1
        idx_to_merge: list[int] = []

        # for head_idx in self._pair_heads[target_pair]:
        #     if head_idx - last_match_idx < 1:
        #         continue
        #     last_match_idx = head_idx
        #     idx_to_merge.append(head_idx)


        pending_merge_cnt = 0
        for idx in range(len(self.tokens) - 1):
            if idx - last_match_idx < 1:
                continue
            pair = self.tokens[idx], self.tokens[idx + 1]
            if pair == target_pair:
                last_match_idx = idx
                idx_to_merge.append(idx - pending_merge_cnt)
                pending_merge_cnt += 1

        # print(f"Found {len(idx_to_merge)} pairs to merge: {idx_to_merge}")
        diff: dict[BytesPair, int] = defaultdict(int)
        for idx, merge_idx in enumerate(idx_to_merge):
            t1 = self.tokens[merge_idx]
            self.tokens[merge_idx] = new_token
            # self._pair_heads[target_pair].appendleft(merge_idx)
            t2 = self.tokens.pop(merge_idx + 1)

            # look backward
            if merge_idx > 0:
                new_pair = (self.tokens[merge_idx - 1], new_token)
                diff[new_pair] += 1
                # self._pair_heads[new_pair].appendleft(merge_idx - 1)

                # previous token is not updated
                if not self.tokens[merge_idx - 1] == new_token:
                    old_pair = (self.tokens[merge_idx - 1], t1)
                    diff[old_pair] -= 1
                # self._pair_heads[old_pair].remove(merge_idx - 1)

            # look forward
            if merge_idx + 1 < len(self.tokens):
                # next token is not going to be updated
                if idx == len(idx_to_merge) - 1 or idx_to_merge[idx + 1] != merge_idx + 1:
                    new_pair = (new_token, self.tokens[merge_idx + 1])
                    diff[new_pair] += 1

                old_pair = (t2, self.tokens[merge_idx + 1])
                diff[old_pair] -= 1
                # self._pair_heads[old_pair].remove(merge_idx + 1)

        # del self._pair_heads[target_pair]

        return diff


class TokenChunk:

    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def pretoken_iter(self):
        """Returns an iterator over the pre-tokenized chunks."""
        return re.finditer(TokenChunk.PAT, self._chunk)

    def __init__(self, chunk: str, use_v2: bool = False):
        pretoken_cls = PreTokenV2 if use_v2 else PreToken
        self._chunk = chunk
        self._pretoken_cnt: dict[PreToken | PreTokenV2, int] = defaultdict(int)

        for pretoken_match in self.pretoken_iter():
            self._pretoken_cnt[pretoken_cls(bytes(pretoken_match.group(), ENCODING))] += 1

    @property
    def pairs_cnt(self) -> dict[BytesPair, int]:
        # return self._pairs_cnt
        pairs_cnt: dict[BytesPair, int] = defaultdict(int)
        for pretoken, pretoken_cnt in self._pretoken_cnt.items():
            # logger.info("Processing pretoken: %s, count: %s", pretoken.value, pretoken_cnt)
            # logger.debug("PreToken pairs cnt: %s", pretoken.pairs_cnt)
            for pair, pair_cnt in pretoken.pairs_cnt.items():
                pairs_cnt[pair] += pair_cnt * pretoken_cnt

        return pairs_cnt

        # logger.info("Pre-tokenized %s unique tokens in chunk.", len(self._pretoken_cnt))
        # logger.debug("Pretoken cnt:\n%s", self._pretoken_cnt)

    def merge_bytes_pair(self, pair: BytesPair) -> dict[BytesPair, int]:
        # logger.debug("Merging pair: %s in TokenChunk", pair)
        chunk_lvl_diff = defaultdict(int)
        diffs: list[tuple[dict[BytesPair, int], int]] = [
            (p.merge_bytes_pair(pair), p_cnt) for p, p_cnt in self._pretoken_cnt.items()
        ]

        for diff, p_cnt in diffs:
            for pair, diff_cnt in diff.items():
                chunk_lvl_diff[pair] += diff_cnt * p_cnt

        return chunk_lvl_diff


class BytePairEncodingBuilder:

    def _split_corpus(self, corpus: str, separators: list[str]) -> list[str]:
        """Splits the corpus into tokens based on the provided separators."""
        regex_pattern = "|".join(map(re.escape, separators))
        return re.split(regex_pattern, corpus)

    def __init__(
        self,
        corpus: str,
        target_vocab_size: int,
        special_tokens: list[str],
        max_merges=-1,
        use_v2: bool = False,
    ):
        self._vocab = [v.to_bytes() for v in range(256)]
        self._target_vocab_size = target_vocab_size
        self._special_tokens = [bytes(t, ENCODING) for t in special_tokens]
        self._max_merges = max_merges

        self._token_chunks: list[TokenChunk] = list(
            map(functools.partial(TokenChunk, use_v2=use_v2), self._split_corpus(corpus, special_tokens))
        )
        self._merges = []
        self._pairs_cnt: dict[BytesPair, int] = defaultdict(int)
        for chunk in self._token_chunks:
            for pair, cnt in chunk.pairs_cnt.items():
                self._pairs_cnt[pair] += cnt

    @property
    def vocab(self) -> dict[int, bytes]:
        """Returns the current vocabulary as a dictionary mapping indices to tokens."""
        return {i: token for i, token in enumerate(self._vocab)} | {
            i: token
            for i, token in enumerate(self._special_tokens, start=len(self._vocab))
        }

    @property
    def merges(self) -> list[BytesPair]:
        """Returns the list of merges that have been applied."""
        return self._merges

    @property
    def vocab_size(self) -> int:
        """Returns the current vocabulary size."""
        return len(self._vocab) + len(self._special_tokens)

    def train(self):

        it = 0
        while True:
            if self._max_merges > 0 and len(self._merges) >= self._max_merges:
                break

            # logger.debug("============================ %02d ============================", it)
            it += 1
            if not self._pairs_cnt:
                # logger.info("No more pairs to merge.")
                break
            target_pair = max(
                self._pairs_cnt, key=(lambda k: (self._pairs_cnt.get(k), k))
            )

            new_token = target_pair[0] + target_pair[1]
            logger.info("New token: %s, cnt: %s", new_token, self._pairs_cnt[target_pair])
            self._vocab.append(new_token)
            self._merges.append(target_pair)

            if self.vocab_size >= self._target_vocab_size:
                break

            if ENALBE_MULTI_PROCESSING and len(self._token_chunks) >= 4 :
                binded_merge_task = functools.partial(merge_task, target_pair)
                with ProcessPoolExecutor(max_workers = min(32, len(self._token_chunks))) as p:
                    diffs_and_chunk = p.map(binded_merge_task, self._token_chunks)
                diffs, self._token_chunks = [], []
                for diff, chunk in diffs_and_chunk:
                    diffs.append(diff)
                    self._token_chunks.append(chunk)
            else:
                diffs = [
                    chunk.merge_bytes_pair(target_pair) for chunk in self._token_chunks
                ]

            del self._pairs_cnt[target_pair]
            for diff in diffs:
                for pair, diff_cnt in diff.items():
                    self._pairs_cnt[pair] += diff_cnt

def merge_task(pair: BytesPair, token_chunk: TokenChunk) -> tuple[dict[BytesPair, int], TokenChunk]:
    diff = token_chunk.merge_bytes_pair(pair)
    return diff, token_chunk

if __name__ == "__main__":
    # logging.basicConfig(level=logging.CRITICAL)
    logging.basicConfig(level=logging.DEBUG)
    #     corpus = """low low low low low
    # lower lower widest widest widest
    # newest newest newest newest newest newest
    # """

    p = PreToken(b' the')
    print("tokens: ", p.tokens)
    diff = p.merge_bytes_pair((b' ', b't'))
    print("diff:", diff)
    print("tokens: ", p.tokens)
    diff = p.merge_bytes_pair((b'h', b'e'))
    print("diff:", diff)
    print("tokens: ", p.tokens)
    diff = p.merge_bytes_pair((b' t', b'h'))
    print("diff:", diff)
    print("tokens: ", p.tokens)

    print("=====================")

    p = PreTokenV2(b' the')
    print("tokens: ", p.tokens)
    diff = p.merge_bytes_pair((b' ', b't'))
    print("diff:", diff)
    print("tokens: ", p.tokens)
    diff = p.merge_bytes_pair((b'h', b'e'))
    print("diff:", diff)
    print("tokens: ", p.tokens)
    diff = p.merge_bytes_pair((b' t', b'h'))
    print("diff:", diff)
    print("tokens: ", p.tokens)

    # import os
    # import time

    # f = open(
    #     os.path.join(
    #         "/home/yq/learning/SF_CS_336/assignment1-basics/tests/fixtures", "corpus.en"
    #     ),
    #     "r",
    #     encoding=ENCODING,
    # )
    # corpus = f.read()
    # f.close()
    # builder = BytePairEncodingBuilder(
    #     corpus, target_vocab_size=500, special_tokens=["<|endoftext|>"]
    # )
    # v1_start = time.time()
    # builder.train()
    # v1_dur = time.time() - v1_start
    # print(f"V1 Duration: {v1_dur:.2f}s")

    # builder = BytePairEncodingBuilder(
    #     corpus, target_vocab_size=500, special_tokens=["<|endoftext|>"], use_v2=True, max_merges=6
    # )
    # v2_start = time.time()
    # builder.train()
    # v2_dur = time.time() - v2_start
    # print(f"V2 Duration: {v2_dur:.2f}s")

    # print("V1 wins!" if v1_dur < v2_dur else "V2 wins!")
    # print(f"{len(builder.vocab)=}")
    # print(f"{builder.vocab_size=}")
    # # print(f"Vocabulary: {builder.vocab}")
    # print(f"Merges: {builder.merges}")
