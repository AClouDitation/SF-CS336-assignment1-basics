#ifndef TOKEN_COLLECTION_H
#define TOKEN_COLLECTION_H

#include <boost/functional/hash.hpp>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bpe {

class TokenCollection {
public:
  using TokenId = size_t;
  using Index = int32_t; 
  using TokenIdPair = std::pair<TokenId, TokenId>;
  using PairFreqMap =
      std::unordered_map<TokenIdPair, int32_t, boost::hash<TokenIdPair>>;

  explicit TokenCollection(std::string_view text);
  ~TokenCollection() = default;

  PairFreqMap GetPairFreq() const;

  // Merges the target pair and returns the diff in pair frequencies.
  PairFreqMap MergePair(const TokenIdPair& target_pair, TokenId new_token_id);

private:
  void ReplacePair(Index first_idx, TokenId new_token_id);

  std::vector<TokenId> token_ids;
  std::vector<Index> prev;
  std::vector<Index> next;
  std::unordered_map<TokenIdPair, std::set<Index>, boost::hash<TokenIdPair>> pair_first_idx;
};

} // namespace bpe

#endif // TOKEN_COLLECTION_H
