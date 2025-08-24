#ifndef BPE_BUILDER_H
#define BPE_BUILDER_H

#include "token_collection.h"

#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace bpe {

class BPEBuilder {
public:
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
  std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>>
  ProcessShard(size_t shard, TokenCollection::TokenIdPair target_pair,
               const std::unordered_map<std::string, size_t> &pretoken_shards);

  const size_t target_vocab_size;
  std::vector<std::string> vocab;
  std::unordered_map<std::string, int32_t> pretoken_freq;
  std::unordered_map<std::string, TokenCollection> token_collections;

  size_t max_total_shards;
  size_t target_pretoken_per_shard;
  std::hash<std::string> hasher;

  TokenCollection::PairFreqMap pairs_cnt;
  std::vector<std::pair<std::string, std::string>> merges;
};

} // namespace bpe

#endif // BPE_BUILDER_H
