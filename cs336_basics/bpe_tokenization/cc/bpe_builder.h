#ifndef BPE_BUILDER_H
#define BPE_BUILDER_H

#include "token_collection.h"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bpe {

class BPEBuilder {
public:
  BPEBuilder(const std::vector<std::string> &special_tokens,
             size_t target_vocab_size);
  ~BPEBuilder() = default;

  void AddPretoken(std::string_view pretoken, int32_t count);
  void Train();

  std::vector<std::pair<std::string, std::string>> GetMerges() const;
  std::vector<std::string> GetVocab() const;

private:
  const size_t target_vocab_size;
  std::vector<std::string> vocab;
  std::unordered_map<std::string, int32_t> pretoken_freq;
  std::unordered_map<std::string, TokenCollection> token_collections;

  TokenCollection::PairFreqMap pairs_cnt;
  std::vector<std::pair<std::string, std::string>> merges;
};

} // namespace bpe

#endif // BPE_BUILDER_H
