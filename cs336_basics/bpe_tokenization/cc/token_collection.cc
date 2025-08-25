#include "token_collection.h"
#include <iostream>

namespace bpe {

TokenCollection::TokenCollection(std::string_view text) : value(text) {
  size_t text_length = text.length();
  assert(text_length <= std::numeric_limits<Index>::max());

  token_ids.reserve(text_length);
  prev.reserve(text_length);
  next.reserve(text_length);

  for (size_t i = 0; i < text_length ; ++i) {
    token_ids.push_back(reinterpret_cast<const unsigned char&>(text[i]));
    prev.push_back(i - 1);
    next.push_back(i + 1);
  }
  next[text_length - 1] = -1;

  for (size_t i = 0; i < text_length - 1; ++i) {
    TokenIdPair pair = std::make_pair(token_ids[i], token_ids[i + 1]);
    pair_first_idx[pair].insert(i);
  }
}

TokenCollection::PairFreqMap TokenCollection::GetPairFreq() const {
  PairFreqMap pair_freq;
  for (const auto &[pair, indices] : pair_first_idx) {
    pair_freq[pair] = indices.size();
  }
  return pair_freq;
}

TokenCollection::PairFreqMap
TokenCollection::MergePair(const TokenIdPair& target_pair, TokenId new_token_id) {
  auto it = pair_first_idx.find(target_pair);
  if (it == pair_first_idx.end() || it->second.empty()) {
    return {};
  }

  PairFreqMap diff;
  const auto& [t1, t2] = target_pair;
  Index last_merge_idx = -1;
  std::vector<Index> idx_to_merge;
  for (const Index idx : it->second) {
    if (prev[idx] >= 0 && prev[idx] == last_merge_idx) {
      continue;
    }
    last_merge_idx = idx;
    idx_to_merge.push_back(idx);
  }

  for(Index merge_idx: idx_to_merge) {
    Index t2_idx = next[merge_idx];
    assert(t2_idx != -1);

    ReplacePair(merge_idx, new_token_id);

    if (Index prev_idx = prev[merge_idx]; prev_idx != -1) {
      TokenIdPair new_pair = std::make_pair(token_ids[prev_idx], new_token_id);
      diff[new_pair] += 1;
      pair_first_idx[new_pair].insert(prev_idx);

      TokenIdPair old_pair = std::make_pair(token_ids[prev_idx], t1);
      diff[old_pair] -= 1;
      pair_first_idx[old_pair].erase(prev_idx);
    }

    if (Index next_idx = next[merge_idx]; next_idx != -1) {
      TokenIdPair new_pair = std::make_pair(new_token_id, token_ids[next_idx]);
      diff[new_pair] += 1;
      pair_first_idx[new_pair].insert(merge_idx);

      TokenIdPair old_pair = std::make_pair(t2, token_ids[next_idx]);
      diff[old_pair] -= 1;
      pair_first_idx[old_pair].erase(t2_idx);
    }
  }

  pair_first_idx.erase(target_pair);
  return diff;
}

void TokenCollection::ReplacePair(Index first_idx, TokenId new_token_id) {
  Index second_idx = next[first_idx];
  assert(second_idx != -1);

  token_ids[first_idx] = new_token_id;
  Index new_next = next[second_idx];
  next[first_idx] = new_next;
  if (new_next != -1) {
    prev[new_next] = first_idx;
  }

  next[second_idx] = -1;
  prev[second_idx] = -1;
  token_ids[second_idx] = -1;
}

} // namespace bpe
