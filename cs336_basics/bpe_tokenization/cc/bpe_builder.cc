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
    : target_vocab_size(target_vocab_size),
      token_collections_shards(max_total_shards),
      max_total_shards(max_total_shards),
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
}

void BPEBuilder::Train() {
  size_t total_shards =
      std::min(static_cast<size_t>(std::ceil(
                   pretoken_freq.size() / double(target_pretoken_per_shard))),
               max_total_shards);
  std::cerr << std::endl
            << "Sharding " << pretoken_freq.size() << " pretokens into "
            << total_shards << " shards." << std::endl;
  if (total_shards > 1) {
    for (const auto &[pretoken, _] : pretoken_freq) {
      token_collections_shards[hasher(pretoken) % total_shards].emplace(
          pretoken, TokenCollection(pretoken));
    }
  } else {
    for (const auto &[pretoken, _] : pretoken_freq) {
      token_collections_shards[0].emplace(
          pretoken, TokenCollection(pretoken));
    }
  }

  for (const auto& token_collections : token_collections_shards) {
    for (const auto &[pretoken, collection] : token_collections) {
      for (const auto &[pair, freq] : collection.GetPairFreq()) {
        pairs_cnt[pair] += pretoken_freq.at(pretoken) * freq;
      }
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

    std::vector<std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>>>
        shard_diffs(total_shards);
    if (total_shards > 1) {
      boost::asio::thread_pool pool(total_shards);
      for (size_t shard = 0; shard < total_shards; ++shard) {
        boost::asio::post(
            pool, std::bind(&BPEBuilder::ProcessShard, this, target_pair,
                            std::ref(token_collections_shards[shard]),
                            &shard_diffs[shard]));
      }
      pool.join();
    } else {
      ProcessShard(target_pair, token_collections_shards[0], &shard_diffs[0]);
    }

    TokenCollection::PairFreqMap merged_diff;
    for (auto &shard : shard_diffs) {
      for (const auto &[diff, freq] : shard) {
        for (const auto &[diff_pair, delta] : diff) {
          merged_diff[diff_pair] += delta * freq;
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
  return target_pair;
}

void BPEBuilder::ProcessShard(
    const TokenCollection::TokenIdPair target_pair,
    std::unordered_map<std::string, TokenCollection>& token_collections,
    std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>> *out) {
  for (auto &[pretoken, collection] : token_collections) {
    out->push_back(
        std::make_pair(collection.MergePair(target_pair, vocab.size() - 1),
                       pretoken_freq.at(pretoken)));
  }
}
} // namespace bpe
