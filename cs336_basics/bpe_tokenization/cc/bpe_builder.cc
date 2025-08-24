#include "bpe_builder.h"

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/use_future.hpp>
#include <cmath>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

namespace bpe {

BPEBuilder::BPEBuilder(const std::vector<std::string> &special_tokens,
                       const size_t target_vocab_size, size_t max_total_shards,
                       size_t target_pretoken_per_shard)
    : target_vocab_size(target_vocab_size), max_total_shards(max_total_shards),
      target_pretoken_per_shard(target_pretoken_per_shard) {
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

    if (total_shards > 1) {
      std::vector<std::future<
          std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>>>>
          diff_futures;
      boost::asio::thread_pool pool(total_shards);
      for (size_t shard = 0; shard < total_shards; ++shard) {
        diff_futures.push_back(boost::asio::post(
            pool, boost::asio::use_future(std::bind(&BPEBuilder::ProcessShard,
                                                    this, shard, target_pair,
                                                    pretoken_shards))));
      }
      pool.join();

      for (auto &future : diff_futures) {
        for (const auto &[diff, freq] : future.get()) {
          for (const auto &[diff_pair, delta] : diff) {
            pairs_cnt[diff_pair] += delta * freq;
          }
        }
      }
    } else {
      for (auto &[pretoken, collection] : token_collections) {
        for (const auto &[diff_pair, delta] :
             collection.MergePair(target_pair, vocab.size() - 1)) {
          pairs_cnt[diff_pair] += delta * pretoken_freq.at(pretoken);
        }
      }
    }

    pairs_cnt.erase(target_pair);
  }
}

std::vector<std::pair<std::string, std::string>> BPEBuilder::GetMerges() const {
  return merges;
}

std::vector<std::string> BPEBuilder::GetVocab() const { return vocab; }

std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>>
BPEBuilder::ProcessShard(
    size_t shard, TokenCollection::TokenIdPair target_pair,
    const std::unordered_map<std::string, size_t> &pretoken_shards) {
  std::vector<std::pair<TokenCollection::PairFreqMap, int32_t>> diffs;
  for (auto &[pretoken, collection] : token_collections) {
    if (pretoken_shards.at(pretoken) != shard) {
      continue;
    }
    diffs.push_back(
        std::make_pair(collection.MergePair(target_pair, vocab.size() - 1),
                       pretoken_freq.at(pretoken)));
  }
  return diffs;
}
} // namespace bpe
