#include "token_collection.h"
#include <iostream>

namespace bpe {

TokenCollection::TokenCollection(std::string_view text) {
  size_t text_length = text.length();
  assert(text_length <= std::numeric_limits<Index>::max());

  token_ids_.reserve(text_length);
  prev_.reserve(text_length);
  next_.reserve(text_length);

  for (Index i = 0; i < text_length ; ++i) {
    token_ids_.push_back(reinterpret_cast<const unsigned char&>(text[i]));
    prev_.push_back(i - 1);
    next_.push_back(i + 1);
  }
  next_[text_length - 1] = -1;

  for (Index i = 0; i < text_length - 1; ++i) {
    TokenIdPair pair = std::make_pair(token_ids_[i], token_ids_[i + 1]);
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
    if (prev_[idx] >= 0 && prev_[idx] == last_merge_idx) {
      continue;
    }
    last_merge_idx = idx;
    idx_to_merge.push_back(idx);
  }

  for(Index merge_idx: idx_to_merge) {
    Index t2_idx = next_[merge_idx];
    assert(t2_idx != -1);

    ReplacePair(merge_idx, new_token_id);

    if (Index prev_idx = prev_[merge_idx]; prev_idx != -1) {
      TokenIdPair new_pair = std::make_pair(token_ids_[prev_idx], new_token_id);
      diff[new_pair] += 1;
      pair_first_idx[new_pair].insert(prev_idx);

      TokenIdPair old_pair = std::make_pair(token_ids_[prev_idx], t1);
      diff[old_pair] -= 1;
      pair_first_idx[old_pair].erase(prev_idx);
    }

    if (Index next_idx = next_[merge_idx]; next_idx != -1) {
      TokenIdPair new_pair = std::make_pair(new_token_id, token_ids_[next_idx]);
      diff[new_pair] += 1;
      pair_first_idx[new_pair].insert(merge_idx);

      TokenIdPair old_pair = std::make_pair(t2, token_ids_[next_idx]);
      diff[old_pair] -= 1;
      pair_first_idx[old_pair].erase(t2_idx);
    }
  }

  pair_first_idx.erase(target_pair);
  return diff;
}

void TokenCollection::ReplacePair(Index first_idx, TokenId new_token_id) {
  Index second_idx = next_[first_idx];
  assert(second_idx != -1);

  token_ids_[first_idx] = new_token_id;
  Index new_next = next_[second_idx];
  next_[first_idx] = new_next;
  if (new_next != -1) {
    prev_[new_next] = first_idx;
  }

  // Unnecessary as these will never be accessed again.
  // next[second_idx] = -1;
  // prev[second_idx] = -1;
  // token_ids[second_idx] = -1;
}

} // namespace bpe
