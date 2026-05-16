#include "frolova_s_radix_sort_double/all/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "util/include/util.hpp"

namespace frolova_s_radix_sort_double {

namespace {

inline std::uint64_t DoubleToSortable(std::uint64_t bits) {
  constexpr std::uint64_t kSignBit = 0x8000000000000000ULL;
  return bits ^ ((bits & kSignBit) ? 0xFFFFFFFFFFFFFFFFULL : kSignBit);
}

inline std::uint64_t SortableToDouble(std::uint64_t bits) {
  constexpr std::uint64_t kSignBit = 0x8000000000000000ULL;
  return bits ^ ((bits & kSignBit) ? 0xFFFFFFFFFFFFFFFFULL : kSignBit);
}

void RadixSortDoubles(std::vector<double> &arr) {
  const std::size_t n = arr.size();
  if (n <= 1) {
    return;
  }

  std::vector<std::uint64_t> keys(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t bits;
    std::memcpy(&bits, &arr[i], sizeof(bits));
    keys[i] = DoubleToSortable(bits);
  }

  std::vector<std::uint64_t> temp(n);
  for (int byte = 0; byte < 8; ++byte) {
    std::size_t count[256] = {};
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint8_t digit = (keys[i] >> (8 * byte)) & 0xFF;
      ++count[digit];
    }

    std::size_t total = 0;
    for (int j = 0; j < 256; ++j) {
      const std::size_t c = count[j];
      count[j] = total;
      total += c;
    }

    for (std::size_t i = 0; i < n; ++i) {
      const std::uint8_t digit = (keys[i] >> (8 * byte)) & 0xFF;
      temp[count[digit]++] = keys[i];
    }
    keys.swap(temp);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t bits = SortableToDouble(keys[i]);
    double val;
    std::memcpy(&val, &bits, sizeof(val));
    arr[i] = val;
  }
}

}  // namespace

RadixSortDouble::RadixSortDouble(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RadixSortDouble::ValidationImpl() {
  return true;
}

bool RadixSortDouble::PreProcessingImpl() {
  return true;
}

bool RadixSortDouble::RunImpl() {
  const auto &input = GetInput();
  const std::size_t n = input.size();
  auto &output = GetOutput();
  output.resize(n);

  if (n == 0) {
    return true;
  }
  if (n == 1) {
    output[0] = input[0];
    return true;
  }

  int num_threads = ppc::util::GetNumThreads();
  if (num_threads <= 0) {
    num_threads = 1;
  }

  const std::size_t base_chunk = n / num_threads;
  const std::size_t remainder = n % num_threads;
  std::vector<std::vector<double>> sorted_chunks(num_threads);

#pragma omp parallel for num_threads(num_threads) default(none) \
    shared(n, base_chunk, remainder, input, sorted_chunks, num_threads)
  for (int t = 0; t < num_threads; ++t) {
    const std::size_t start = t * base_chunk + std::min<std::size_t>(t, remainder);
    const std::size_t end = (t + 1) * base_chunk + std::min<std::size_t>(t + 1, remainder);
    if (end == start) {
      continue;
    }

    std::vector<double> chunk(input.begin() + start, input.begin() + end);
    RadixSortDoubles(chunk);
    sorted_chunks[t] = std::move(chunk);
  }

  std::vector<std::pair<std::size_t, std::size_t>> segments;
  std::size_t offset = 0;
  for (int t = 0; t < num_threads; ++t) {
    if (!sorted_chunks[t].empty()) {
      const auto &chunk = sorted_chunks[t];
      std::copy(chunk.begin(), chunk.end(), buffer1.begin() + offset);
      segments.emplace_back(offset, chunk.size());
      offset += chunk.size();
    }
  }

  if (segments.size() <= 1) {
    output = std::move(buffer1);
    return true;
  }

  std::vector<double> buffer2(n);
  while (segments.size() > 1) {
    std::vector<std::pair<std::size_t, std::size_t>> new_segments;
    std::size_t new_offset = 0;
    for (std::size_t i = 0; i < segments.size(); i += 2) {
      const std::size_t len1 = segments[i].second;
      const std::size_t len2 = (i + 1 < segments.size()) ? segments[i + 1].second : 0;
      new_segments.emplace_back(new_offset, len1 + len2);
      new_offset += len1 + len2;
    }

    tbb::parallel_for(std::size_t(0), (segments.size() + 1) / 2, [&](std::size_t pair_idx) {
      const std::size_t i = pair_idx * 2;
      if (i + 1 < segments.size()) {
        const auto &seg1 = segments[i];
        const auto &seg2 = segments[i + 1];
        const auto dest = new_segments[pair_idx].first;
        std::merge(buffer1.begin() + seg1.first, buffer1.begin() + seg1.first + seg1.second,
                   buffer1.begin() + seg2.first, buffer1.begin() + seg2.first + seg2.second, buffer2.begin() + dest);
      } else {
        const auto &seg = segments[i];
        const auto dest = new_segments[pair_idx].first;
        std::copy(buffer1.begin() + seg.first, buffer1.begin() + seg.first + seg.second, buffer2.begin() + dest);
      }
    });

    buffer1.swap(buffer2);
    segments = std::move(new_segments);
  }

  output = std::move(buffer1);
  return true;
}

bool RadixSortDouble::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_radix_sort_double
