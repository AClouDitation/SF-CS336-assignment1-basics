import os
import logging
import regex as re

from collections import defaultdict
from cs336_basics.bpe_tokenization.token_collection import TokenCollection, TokenIdPair
from cs336_basics.bpe_tokenization.vocab import Vocab
from cs336_basics.bpe_tokenization.pretokenization import pretokenize


ENCODING = "utf-8"

logger = logging.getLogger(__name__)

class BytePairEncodingBuilder:
    PAT = re.compile(
        rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def __init__(
        self,
        file_path: str | os.PathLike,
        target_vocab_size: int,
        special_tokens: list[str],
        max_merges=-1,
    ):
        self._vocab = Vocab(special_tokens)
        self._target_vocab_size = target_vocab_size
        self._max_merges = max_merges

        self._pretoken_cnt: dict[bytes, int] = pretokenize(
            file_path, [t.encode(ENCODING) for t in special_tokens]
        )

        self._token_collections = {k: TokenCollection(k) for k in self._pretoken_cnt}

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

            target_pair: TokenIdPair = (-1, -1)
            max_cnt = -1
            for pair, cnt in self._pairs_cnt.items():
                if cnt < max_cnt:
                    continue
                if cnt > max_cnt or (
                    cnt == max_cnt
                    and self._vocab.token_pair(pair)
                    > self._vocab.token_pair(target_pair)
                ):
                    target_pair = pair
                    max_cnt = cnt

            token_pair = self._vocab.token_pair(target_pair)
            new_token = token_pair[0] + token_pair[1]
            new_token_id = self._vocab.add(new_token)
            # logger.info(
            #     "New token: %s, cnt: %s", new_token, self._pairs_cnt[target_pair]
            # )
            self._merges.append(token_pair)

            if self.vocab_size >= self._target_vocab_size:
                break

            diffs = [
                (diff, self._pretoken_cnt[pretoken])
                for pretoken, coll in self._token_collections.items()
                if (diff := coll.merge_bytes_pair(target_pair, new_token_id))
            ]

            irrelevant_pairs: set[TokenIdPair] = {target_pair}
            for diff, multiplier in diffs:
                for pair, diff_cnt in diff.items():
                    self._pairs_cnt[pair] += diff_cnt * multiplier
            for pair, cnt in self._pairs_cnt.items():
                if cnt <= 0:
                    irrelevant_pairs.add(pair)
            for pair in irrelevant_pairs:
                del self._pairs_cnt[pair]
