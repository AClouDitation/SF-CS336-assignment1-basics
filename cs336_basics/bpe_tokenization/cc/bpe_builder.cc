#include "bpe_builder.h"

#include <iostream>

namespace bpe {

BPEBuilder::BPEBuilder(const std::vector<std::string> &special_tokens,
                       const size_t target_vocab_size)
    : target_vocab_size(target_vocab_size) {
  vocab.reserve(sizeof(char) + special_tokens.size());
  for (int i = 0; i < 256; ++i) {
    vocab.push_back(std::string(1, reinterpret_cast<char &>(i)));
  }
  for (const std::string &token : special_tokens) {
    vocab.push_back(token);
  }
}

void BPEBuilder::AddPretoken(std::string_view pretoken, const int32_t count) {
  pretoken_freq.emplace(pretoken, count);
  token_collections.emplace(pretoken, TokenCollection(pretoken));
}

void BPEBuilder::Train() {
  for (const auto &[pretoken, collection] : token_collections) {
    for (const auto &[pair, freq] : collection.GetPairFreq()) {
      pairs_cnt[pair] += pretoken_freq[pretoken] * freq;
    }
  }

  while (true) {
    if (pairs_cnt.empty()) {
      break;
    }

    // Find the most frequent pair
    TokenCollection::TokenIdPair target_pair;
    int32_t max_count = -1;
    for (const auto &[pair, cnt] : pairs_cnt) {
      if (cnt < max_count) {
        continue;
      }
      if (cnt > max_count || vocab[pair.first] > vocab[target_pair.first] ||
          (vocab[pair.first] == vocab[target_pair.first] &&
           vocab[pair.second] > vocab[target_pair.second])) {
        target_pair = pair;
        max_count = cnt;
      }
    }

    vocab.push_back(vocab[target_pair.first] + vocab[target_pair.second]);
    merges.push_back(
        std::make_pair(vocab[target_pair.first], vocab[target_pair.second]));

    if (vocab.size() >= target_vocab_size) {
      break;
    }

    TokenCollection::PairFreqMap merged_diff;
    for (auto &[pretoken, collection] : token_collections) {
      for (const auto &[diff_pair, delta] :
           collection.MergePair(target_pair, vocab.size() - 1)) {
        merged_diff[diff_pair] += delta * pretoken_freq[pretoken];
        pairs_cnt[diff_pair] += delta * pretoken_freq[pretoken];
      }
    }

    pairs_cnt.erase(target_pair);
  }
}

std::vector<std::pair<std::string, std::string>> BPEBuilder::GetMerges() const {
  return merges;
}

std::vector<std::string> BPEBuilder::GetVocab() const { return vocab; }

} // namespace bpe
