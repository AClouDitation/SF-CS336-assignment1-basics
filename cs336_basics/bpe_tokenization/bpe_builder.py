import os
import pathlib
import logging
import regex as re
import cppyy

from collections import defaultdict
from cs336_basics.bpe_tokenization.token_collection import TokenCollection, TokenIdPair
from cs336_basics.bpe_tokenization.vocab import Vocab


ENCODING = "utf-8"

logger = logging.getLogger(__name__)

CC_PATH = (pathlib.Path(__file__).resolve().parent) / "cc"
# cppyy.add_include_path(str(CC_PATH))
# cppyy.cppdef(open(CC_PATH / "token_collection.cc").read())
# cppyy.cppdef(open(CC_PATH / "bpe_builder.cc").read())
cppyy.include(str(CC_PATH / "token_collection.h"))
cppyy.load_library(str(CC_PATH / "libtoken_collection.so"))
cppyy.include(str(CC_PATH / "bpe_builder.h"))
cppyy.load_library(str(CC_PATH / "libbpe_builder.so"))
from cppyy.gbl import bpe as cc # type: ignore
cc.BPEBuilder.Train.__release_gil__ = True

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

            del self._pairs_cnt[target_pair]
            for diff, multiplier in diffs:
                for pair, diff_cnt in diff.items():
                    self._pairs_cnt[pair] += diff_cnt * multiplier


def train_bpe(
    corpus: str,
    target_vocab_size: int,
    special_tokens: list[str],
    use_cpp: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if use_cpp:
        regex_pattern = "|".join(map(re.escape, special_tokens))
        pretoken_cnt: dict[str, int] = defaultdict(int)
        for chunk in re.split(regex_pattern, corpus):
            for pretoken in re.finditer(BytePairEncodingBuilder.PAT, chunk):
                pretoken_cnt[pretoken.group()] += 1

        builder = cc.BPEBuilder(
            special_tokens=special_tokens,
            target_vocab_size=target_vocab_size
        )
        for pretoken, count in pretoken_cnt.items():
            builder.AddPretoken(pretoken, count)

        builder.Train()
        vocab = builder.GetVocab()
        merges = builder.GetMerges()

        return {i: bytes(t) for i, t in enumerate(vocab)}, [
            (merge.first, merge.second) for merge in merges
        ]  # type:ignore

    builder = BytePairEncodingBuilder(corpus, target_vocab_size, special_tokens)
    builder.train()

    return builder.vocab, builder.merges


if __name__ == "__main__":
    # logging.basicConfig(level=logging.CRITICAL)
    logging.basicConfig(level=logging.INFO)

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

    start = time.time()
    train_bpe(corpus, 500, ["<|endoftext|>"], use_cpp=True)
    dur = time.time() - start
    print(f"CPP Duration: {dur:.2f}s")

    start = time.time()
    train_bpe(corpus, 500, ["<|endoftext|>"])
    dur = time.time() - start
    print(f"Py Duration: {dur:.2f}s")
