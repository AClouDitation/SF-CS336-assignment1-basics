import logging

from typing import TypeAlias
from collections import defaultdict, OrderedDict

TokenIdPair: TypeAlias = tuple[int, int]

logger = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)

# def info(type, value, tb):
#     if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type != AssertionError:
#         # we are in interactive mode or we don't have a tty-like
#         # device, so we call the default hook
#         sys.__excepthook__(type, value, tb)
#     else:
#         import traceback, pdb
#         # we are NOT in interactive mode, print the exception...
#         traceback.print_exception(type, value, tb)
#         print
#         # ...then start the debugger in post-mortem mode.
#         pdb.pm()

# sys.excepthook = info

class TokenCollection:
    def __init__(self, value: bytes):
        self.token_ids = [b for b in value]
        self.next = [i for i in range(1, len(self.token_ids))] + [None]
        self.prev = [None] + [i for i in range(len(self.token_ids) - 1)]
        self._pair_first_idx: dict[TokenIdPair, OrderedDict[int, None]] = defaultdict(
            OrderedDict[int, None]
        )

        for i in range(len(self.token_ids) - 1):
            self._pair_first_idx[(self.token_ids[i], self.token_ids[i + 1])][i] = None

    @property
    def pairs_cnt(self) -> dict[TokenIdPair, int]:
        pairs_cnt: dict[TokenIdPair, int] = defaultdict(int)
        for idx in range(len(self.token_ids) - 1):
            pair = self.token_ids[idx], self.token_ids[idx + 1]
            pairs_cnt[pair] += 1
        return pairs_cnt

    def _replace_pair(self, first_idx: int, new_token_id: int):
        second_idx = self.next[first_idx]
        assert second_idx is not None

        self.token_ids[first_idx] = new_token_id
        new_next = self.next[second_idx]
        self.next[first_idx] = self.next[second_idx]
        if new_next:
            self.prev[new_next] = first_idx

        self.next[second_idx] = None
        self.prev[second_idx] = None
        self.token_ids[second_idx] = -1  # mark as removed

    def merge_bytes_pair(
        self, target_pair: TokenIdPair, new_token_id: int
    ) -> dict[TokenIdPair, int]:
        if (
            target_pair not in self._pair_first_idx
            or len(self._pair_first_idx[target_pair]) == 0
        ):
            return {}

        last_match_idx = None
        idx_to_merge: list[int] = []
        for idx in self._pair_first_idx[target_pair]:
            if self.prev[idx] is not None and self.prev[idx] == last_match_idx:
                continue
            last_match_idx = idx
            idx_to_merge.append(idx)

        # logger.debug("found %d occurrences", len(idx_to_merge))
        diff: dict[TokenIdPair, int] = defaultdict(int)
        t1, t2 = target_pair
        for merge_idx in idx_to_merge:
            t2_idx = self.next[merge_idx]
            assert t2_idx is not None, "Next index is None, cannot merge pair"
            # logger.debug("Merging tokens at idx: %d and idx: %d", t1_idx, t2_idx)
            # logger.debug("prev: %s, next: %s", self.prev[t1_idx], self.next[t2_idx])

            self._replace_pair(merge_idx, new_token_id)

            # look backward
            if (prev_idx := self.prev[merge_idx]) is not None:
                new_pair = (self.token_ids[prev_idx], new_token_id)
                diff[new_pair] += 1
                self._pair_first_idx[new_pair][prev_idx] = None

                old_pair = (self.token_ids[prev_idx], t1)
                diff[old_pair] -= 1
                del self._pair_first_idx[old_pair][prev_idx]

            # look forward
            if (next_idx := self.next[merge_idx]) is not None:
                new_pair = (new_token_id, self.token_ids[next_idx])
                diff[new_pair] += 1
                self._pair_first_idx[new_pair][merge_idx] = None

                old_pair = (t2, self.token_ids[next_idx])
                diff[old_pair] -= 1
                del self._pair_first_idx[old_pair][t2_idx]

        del self._pair_first_idx[target_pair]
        return diff
