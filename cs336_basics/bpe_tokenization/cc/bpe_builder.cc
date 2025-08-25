#include "bpe_builder.h"

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/progress.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>

namespace bpe {

BPEBuilder::BPEBuilder(const std::vector<std::string> &special_tokens,
                       const size_t target_vocab_size, size_t max_total_shards,
                       size_t target_pretoken_per_shard)
    : target_vocab_size(target_vocab_size), max_total_shards(max_total_shards),
      target_pretoken_per_shard(target_pretoken_per_shard) {
  std::function<bool(const FreqPair &, const FreqPair &)> comparator =
      std::bind_front(&BPEBuilder::FreqPairComparator, this);
  pairs_cnt_queue = decltype(pairs_cnt_queue)(comparator);

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
  size_t total_shards =
      std::min(static_cast<size_t>(std::ceil(
                   pretoken_freq.size() / double(target_pretoken_per_shard))),
               max_total_shards);
  std::cerr << std::endl
            << "Sharding " << pretoken_freq.size() << " pretokens into "
            << total_shards << " shards." << std::endl;

  std::unordered_map<std::string, size_t> pretoken_shards;
  if (total_shards > 1) {
    for (const auto &[pretoken, freq] : pretoken_freq) {
      pretoken_shards[pretoken] = hasher(pretoken) % total_shards;
    }
  }

  for (const auto &[pretoken, collection] : token_collections) {
    for (const auto &[pair, freq] : collection.GetPairFreq()) {
      pairs_cnt[pair] += pretoken_freq.at(pretoken) * freq;
    }
  }
  for (const auto &[pair, cnt] : pairs_cnt) {
    pairs_cnt_queue.push(std::make_pair(cnt, pair));
  }

  boost::progress_display progress(target_vocab_size - vocab.size());
  while (true) {
    if (pairs_cnt.empty()) {
      break;
    }

    // Find the most frequent pair
    TokenCollection::TokenIdPair target_pair = FindBestPair();
    vocab.push_back(vocab[target_pair.first] + vocab[target_pair.second]);
    merges.push_back(
        std::make_pair(vocab[target_pair.first], vocab[target_pair.second]));

    if (vocab.size() >= target_vocab_size) {
      break;
    }

    TokenCollection::PairFreqMap merged_diff;
    if (total_shards > 1) {
      std::vector<std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>>>
          shard_diffs;
      shard_diffs.reserve(total_shards);
      for (size_t i = 0; i < total_shards; ++i) {
        shard_diffs.push_back({});
      }

      boost::asio::thread_pool pool(total_shards);
      for (size_t shard = 0; shard < total_shards; ++shard) {
        boost::asio::post(pool, std::bind(&BPEBuilder::ProcessShard, this,
                                          shard, target_pair, pretoken_shards,
                                          &shard_diffs[shard]));
      }
      pool.join();

      for (auto &shard : shard_diffs) {
        for (const auto &[diff, freq] : shard) {
          for (const auto &[diff_pair, delta] : diff) {
            merged_diff[diff_pair] += delta * freq;
          }
        }
      }
    } else {
      for (auto &[pretoken, collection] : token_collections) {
        for (const auto &[diff_pair, delta] :
             collection.MergePair(target_pair, vocab.size() - 1)) {
          merged_diff[diff_pair] += delta * pretoken_freq.at(pretoken);
        }
      }
    }

    std::vector<TokenCollection::TokenIdPair> pairs_to_delete;
    for (const auto &[diff_pair, delta] : merged_diff) {
      pairs_cnt[diff_pair] += delta;
      if (pairs_cnt[diff_pair] <= 0) {
        pairs_to_delete.push_back(diff_pair);
      } else {
        pairs_cnt_queue.push(std::make_pair(pairs_cnt[diff_pair], diff_pair));
      }
    }

    for (const auto &pair : pairs_to_delete) {
      pairs_cnt.erase(pair);
    }
    pairs_cnt.erase(target_pair);
    ++progress;
  }
  std::cout << std::endl;
}

std::vector<std::pair<std::string, std::string>> BPEBuilder::GetMerges() const {
  return merges;
}

std::vector<std::string> BPEBuilder::GetVocab() const { return vocab; }

bool BPEBuilder::FreqPairComparator(const FreqPair &lhs,
                                    const FreqPair &rhs) const {
  if (lhs.first != rhs.first) {
    return lhs.first < rhs.first;
  }
  std::string_view lt1 = vocab[lhs.second.first];
  std::string_view rt1 = vocab[rhs.second.first];
  if (lt1 != rt1) {
    return lt1 < rt1;
  }
  std::string_view lt2 = vocab[lhs.second.second];
  std::string_view rt2 = vocab[rhs.second.second];
  return lt2 < rt2;
}

TokenCollection::TokenIdPair BPEBuilder::FindBestPair() {
  TokenCollection::TokenIdPair target_pair;

  while (!pairs_cnt_queue.empty()) {
    const auto &[count, pair] = pairs_cnt_queue.top();
    auto it = pairs_cnt.find(pair);
    if (it != pairs_cnt.end() && it->second == count) {
      target_pair = pair;
      break;
    }
    pairs_cnt_queue.pop();
  }
  // int32_t max_count = -1;
  // for (const auto &[pair, cnt] : pairs_cnt) {
  //   if (FreqPairComparator(std::make_pair(cnt, pair),
  //                          std::make_pair(max_count, target_pair))) {
  //     target_pair = pair;
  //     max_count = cnt;
  //   }
  // }
  return target_pair;
}

void BPEBuilder::ProcessShard(
    size_t shard, TokenCollection::TokenIdPair target_pair,
    const std::unordered_map<std::string, size_t> &pretoken_shards,
    std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>> *out) {
  for (auto &[pretoken, collection] : token_collections) {
    if (pretoken_shards.at(pretoken) != shard) {
      continue;
    }
    out->push_back(
        std::make_pair(collection.MergePair(target_pair, vocab.size() - 1),
                       pretoken_freq.at(pretoken)));
  }
}
} // namespace bpe
