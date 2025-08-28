#include "bpe_builder.h"

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/progress.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <thread>

namespace bpe {
namespace {

using Sharder = std::function<size_t(const std::string &)>;

size_t PretokenSharder(const size_t total_shards, const std::string &pretoken) {
  const static std::hash<std::string> hasher;
  return hasher(pretoken) % total_shards;
}

size_t SingleShardSharder(const std::string &pretoken) {
  return 0ul;
}

} // namespace

BPEBuilder::BPEBuilder(const std::vector<std::string> &special_tokens,
                       const size_t target_vocab_size, size_t max_total_shards,
                       size_t target_pretoken_per_shard)
    : target_vocab_size_(target_vocab_size),
      token_collections_shards_(max_total_shards),
      max_total_shards_(max_total_shards),
      target_pretoken_per_shard_(target_pretoken_per_shard) {
  std::function<bool(const FreqPair &, const FreqPair &)> comparator =
      std::bind_front(&BPEBuilder::FreqPairComparator, this);
  pairs_cnt_queue_ = decltype(pairs_cnt_queue_)(comparator);

  vocab_.reserve(sizeof(char) + special_tokens.size());
  for (int i = 0; i < 256; ++i) {
    vocab_.push_back(std::string(1, reinterpret_cast<char &>(i)));
  }
  for (const std::string &token : special_tokens) {
    vocab_.push_back(token);
  }
}

void BPEBuilder::AddPretoken(std::string_view pretoken, const int32_t count) {
  pretoken_freq_.emplace(pretoken, count);
}

void BPEBuilder::Train() {
  size_t total_shards =
      std::min(static_cast<size_t>(std::ceil(
                   pretoken_freq_.size() / double(target_pretoken_per_shard_))),
               max_total_shards_);
  std::cerr << std::endl
            << "Sharding " << pretoken_freq_.size() << " pretokens into "
            << total_shards << " shards." << std::endl;
  std::function<size_t(const std::string &)> sharder =
      total_shards > 1 ? Sharder(std::bind_front(&PretokenSharder, total_shards))
                       : &SingleShardSharder;
  for (const auto &[pretoken, freq] : pretoken_freq_) {
    token_collections_shards_[sharder(pretoken)].emplace_back(
        TokenCollection(pretoken), freq);
  }

  for (const auto& token_collections : token_collections_shards_) {
    for (const auto &[collection, pretoken_freq] : token_collections) {
      for (const auto &[pair, pair_freq] : collection.GetPairFreq()) {
        pairs_cnt_[pair] += pretoken_freq * pair_freq;
      }
    }
  }
  for (const auto &freq_pair : pairs_cnt_) {
    pairs_cnt_queue_.push(freq_pair);
  }

  boost::progress_display progress(target_vocab_size_ - vocab_.size());
  while (true) {
    if (pairs_cnt_.empty()) {
      break;
    }

    // Find the most frequent pair
    TokenCollection::TokenIdPair target_pair = FindBestPair();
    vocab_.push_back(vocab_[target_pair.first] + vocab_[target_pair.second]);
    merges_.push_back(
        std::make_pair(vocab_[target_pair.first], vocab_[target_pair.second]));

    if (vocab_.size() >= target_vocab_size_) {
      break;
    }

    std::vector<std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>>>
        shard_diffs(total_shards);
    if (total_shards > 1) {
      boost::asio::thread_pool pool(total_shards);
      for (size_t shard = 0; shard < total_shards; ++shard) {
        boost::asio::post(
            pool, std::bind(&BPEBuilder::ProcessShard, this, target_pair,
                            std::ref(token_collections_shards_[shard]),
                            &shard_diffs[shard]));
      }
      pool.join();
    } else {
      ProcessShard(target_pair, token_collections_shards_[0], &shard_diffs[0]);
    }

    std::vector<TokenCollection::TokenIdPair> pairs_to_delete{target_pair};
    std::unordered_set<TokenCollection::TokenIdPair,
                       boost::hash<TokenCollection::TokenIdPair>>
        updated_pairs;
    for (const auto &shard : shard_diffs) {
      for (const auto &[diff, freq] : shard) {
        for (const auto &[diff_pair, delta] : diff) {
          pairs_cnt_[diff_pair] += delta * freq;
          if (pairs_cnt_[diff_pair] <= 0) {
            pairs_to_delete.push_back(diff_pair);
          } else {
            updated_pairs.insert(diff_pair);
          }
        }
      }
    }

    for(const auto& pair: updated_pairs) {
      pairs_cnt_queue_.push(std::make_pair(pair, pairs_cnt_.at(pair)));
    }

    for (const auto &pair : pairs_to_delete) {
      pairs_cnt_.erase(pair);
    }
    ++progress;
  }
  std::cout << std::endl;
}

std::vector<std::pair<std::string, std::string>> BPEBuilder::GetMerges() const {
  return merges_;
}

std::vector<std::string> BPEBuilder::GetVocab() const { return vocab_; }

bool BPEBuilder::FreqPairComparator(const FreqPair &lhs,
                                    const FreqPair &rhs) const {
  if (lhs.second != rhs.second) {
    return lhs.second < rhs.second;
  }
  std::string_view lt1 = vocab_[lhs.first.first];
  std::string_view rt1 = vocab_[rhs.first.first];
  if (lt1 != rt1) {
    return lt1 < rt1;
  }
  std::string_view lt2 = vocab_[lhs.first.second];
  std::string_view rt2 = vocab_[rhs.first.second];
  return lt2 < rt2;
}

TokenCollection::TokenIdPair BPEBuilder::FindBestPair() {
  TokenCollection::TokenIdPair target_pair;

  while (!pairs_cnt_queue_.empty()) {
    const auto &[pair, count] = pairs_cnt_queue_.top();
    auto it = pairs_cnt_.find(pair);
    if (it != pairs_cnt_.end() && it->second == count) {
      target_pair = pair;
      break;
    }
    pairs_cnt_queue_.pop();
  }
  return target_pair;
}

void BPEBuilder::ProcessShard(
    const TokenCollection::TokenIdPair target_pair,
    std::vector<std::pair<TokenCollection, int32_t>>& token_collections,
    std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>> *out) {
  for (auto &[collection, freq] : token_collections) {
    out->push_back(std::make_pair(
        collection.MergePair(target_pair, vocab_.size() - 1), freq));
  }
}
} // namespace bpe
