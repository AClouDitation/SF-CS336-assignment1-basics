import logging
import regex as re

from pretoken import TokenCollection, TokenIdPair
from collections import defaultdict
from vocab import Vocab


ENCODING = "utf-8"
ENABLE_MULTI_PROCESSING = False
logger = logging.getLogger(__name__)


class BytePairEncodingBuilder:
    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

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
    ):
        self._vocab = Vocab(special_tokens)
        self._target_vocab_size = target_vocab_size
        self._max_merges = max_merges

        self._pretoken_cnt: dict[str, int] = defaultdict(int)
        for chunk in self._split_corpus(corpus, special_tokens):
            for pretoken in re.finditer(BytePairEncodingBuilder.PAT, chunk):
                self._pretoken_cnt[pretoken.group()] += 1

        self._token_collections = {
            k: TokenCollection(bytes(k, ENCODING)) for k in self._pretoken_cnt
        }

        self._merges = []
        self._pairs_cnt: dict[TokenIdPair, int] = defaultdict(int)
        for pre_token, coll in self._token_collections.items():
            for pair, cnt in coll.pairs_cnt.items():
                self._pairs_cnt[pair] += cnt * self._pretoken_cnt[pre_token]

    @property
    def vocab(self) -> dict[int, bytes]:
        """Returns the current vocabulary as a dictionary mapping indices to tokens."""
        return {i: token for i, token in enumerate(self._vocab._token_by_id)}

    @property
    def merges(self) -> list[tuple[bytes, bytes]]:
        """Returns the list of merges that have been applied."""
        return self._merges

    @property
    def vocab_size(self) -> int:
        """Returns the current vocabulary size."""
        return len(self._vocab)

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
                self._pairs_cnt,
                key=(
                    lambda k: (
                        self._pairs_cnt.get(k),
                        (self._vocab.token(k[0]), self._vocab.token(k[1])),
                    )
                ),
            )

            token_1 = self._vocab.token(target_pair[0])
            token_2 = self._vocab.token(target_pair[1])
            # assert token_1 is not None and token_2 is not None
            new_token = token_1 + token_2 # type: ignore
            new_token_id = self._vocab.add(new_token)
            # logger.info(
            #     "New token: %s, cnt: %s", new_token, self._pairs_cnt[target_pair]
            # )
            self._merges.append((token_1, token_2))

            if self.vocab_size >= self._target_vocab_size:
                break

            diffs = [
                (diff, self._pretoken_cnt[pretoken])
                for pretoken, coll in self._token_collections.items()
                if (diff := coll.merge_bytes_pair(target_pair, new_token_id))
            ]

            del self._pairs_cnt[target_pair]
            for diff, multiplier in diffs:
                for pair, diff_cnt in diff.items():
                    self._pairs_cnt[pair] += diff_cnt * multiplier


if __name__ == "__main__":
    # logging.basicConfig(level=logging.CRITICAL)
    logging.basicConfig(level=logging.INFO)

    # vocab = Vocab(special_tokens=[])
    # def print_tokens(token_ids: list[int]):
    #     print('tokens:', [vocab.token(tid) for tid in token_ids])

    # def print_diff(diff: dict[TokenIdPair, int]):
    #     token_diff = {
    #         (vocab.token(pair[0]), vocab.token(pair[1])): count
    #         for pair, count in diff.items()
    #     }
    #     print('diff:', token_diff)

    # p = TokenCollection(b" the")
    # print_tokens(p.token_ids)
    # print()

    # for pair in [(b' ', b't'), (b't', b'h'), (b'h', b'e')]:
    #     print("merging ", pair)
    #     new_token_id = vocab.add(pair[0] + pair[1])
    #     diff = p.merge_bytes_pair(
    #         (vocab.token_id(pair[0]), vocab.token_id(pair[1])), new_token_id
    #     )
    #     print_diff(diff)
    #     print_tokens(p.token_ids)
    #     print("next: ", p.next)
    #     print("prev: ", p.prev)
    #     print(
    #         "pf: ",
    #         {
    #             (vocab.token(k[0]), vocab.token(k[1])): [idx for idx in v]
    #             for k, v in p._pair_first_idx.items()
    #         },
    #     )
    #     print()

    import os
    import time

    f = open(
        os.path.join(
            "/home/yq/learning/SF_CS_336/assignment1-basics/tests/fixtures", "corpus.en"
        ),
        "r",
        encoding=ENCODING,
    )
    corpus = f.read()
    f.close()
    builder = BytePairEncodingBuilder(
        corpus, target_vocab_size=500, special_tokens=["<|endoftext|>"]
    )
    start = time.time()
    builder.train()
    dur = time.time() - start
    print(f"Duration: {dur:.2f}s")
