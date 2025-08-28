#ifndef BPE_BUILDER_H
#define BPE_BUILDER_H

#include "token_collection.h"

#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bpe {

class BPEBuilder {
public:
  BPEBuilder(const BPEBuilder&) = delete;
  BPEBuilder& operator=(const BPEBuilder&) = delete;
  BPEBuilder(const std::vector<std::string> &special_tokens,
             size_t target_vocab_size,
             size_t max_total_shards = std::thread::hardware_concurrency(),
             size_t target_pretoken_per_shard = 10000);
  ~BPEBuilder() = default;

  void AddPretoken(std::string_view pretoken, int32_t count);
  void Train();

  std::vector<std::pair<std::string, std::string>> GetMerges() const;
  std::vector<std::string> GetVocab() const;

private:
  TokenCollection::TokenIdPair FindBestPair();

  using FreqPair = std::pair<TokenCollection::TokenIdPair, int32_t>;
  bool FreqPairComparator(const FreqPair &lhs, const FreqPair &rhs) const;

  void ProcessShard(
      TokenCollection::TokenIdPair target_pair,
      std::vector<std::pair<TokenCollection, int32_t>> &token_collections,
      std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>> *out);

  const size_t target_vocab_size_;
  std::vector<std::string> vocab_;
  std::unordered_map<std::string, int32_t> pretoken_freq_;
  std::vector<std::vector<std::pair<TokenCollection, int32_t>>> token_collections_shards_;

  size_t max_total_shards_;
  size_t target_pretoken_per_shard_;

  TokenCollection::PairFreqMap pairs_cnt_;

  std::priority_queue<FreqPair, std::vector<FreqPair>,
                      std::function<bool(const FreqPair &, const FreqPair &)>>
      pairs_cnt_queue_;
  std::vector<std::pair<std::string, std::string>> merges_;
};

} // namespace bpe

#endif // BPE_BUILDER_H
