#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "i_softmax_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

template <typename T> void fill_seq(T *t, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    int start = i;
    for (size_t j = 0; j < cols; ++j)
      t[i][j] = start++;
  }
}

void softmax_isolation(void *d, void *c, int len, float m1, float m2,
                       int64_t ld);
void softmax_isolation_8(void *d, void *c, int len, float m1, float m2,
                          int64_t ld);
void softmax_isolation_16(void *d, void *c, int len, float m1, float m2,
                          int64_t ld);

static constexpr size_t cols = 384;
static constexpr size_t rows = 16;

int main() {
  int32_t c[24][rows][cols];
  int8_t d[24][rows][cols];
  int8_t dc[25][rows][cols];

  for (int i = 0; i < 24; ++i)
    fill_seq(c[i], rows, cols);

  memset(d, 0, sizeof(d));
  memset(dc, 0, sizeof(dc));

  auto start = Time::now();

  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 24; ++j) {
      softmax_isolation((void *)d[j], (void *)c[j], cols, 0.01, 10000, cols);
    }
  }

  auto during =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
          .count();
  std::cout << "softmax time : " << (float)during / 1000 / 1000
            << " ms " << std::endl;

  start = Time::now();
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 24; ++j) {
      softmax_isolation_16(d[j], c[j], cols, 0.01, 10000, cols);
      softmax_isolation_8(dc[j], c[j], cols, 0.01, 10000, cols);
    }
  }

  during =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
          .count();
  std::cout << "tile softmax time : "
            << (float)during / 1000 / 1000 << " ms " << std::endl;

  if (memcmp(d, dc, sizeof(d)) != 0) {
    std::cout<< "Something happend"<<std::endl;
  }

  start = Time::now();
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 24; ++j) {
      softmax_isolation(d[j], c[j], cols, 0.01, 10000, cols);
    }
  }

  during =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
          .count();
  std::cout << "softmax time : "
            << (float)during / 1000 / 1000 << " ms " << std::endl;

  return 0;
}

void softmax_isolation(void *d, void *c, int len, float m1, float m2,
                       int64_t ld) {
  auto d_ = reinterpret_cast<int8_t(*)[cols]>(d);
  auto c_ = reinterpret_cast<int(*)[cols]>(c);
  intel_mlperf::i32_scale_attlen_softmax_scale_i8<16, 8>::run(d, c, len, m1, m2,
                                                              ld);
  intel_mlperf::i32_scale_attlen_softmax_scale_i8<16, 8>::run(d_ + 8, c_ + 8,
                                                              len, m1, m2, ld);
}

void softmax_isolation_8(void *d, void *c, int len, float m1, float m2, int64_t ld) {
  auto d_ = reinterpret_cast<int8_t (*)[64]>(d);
  auto c_ = reinterpret_cast<int (*)[16]>(c);
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<8>::run(d_, c_, len, m1, m2);
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<8>::run(d_ + 8, c_ + 8, len, m1, m2);
}

void softmax_isolation_16(void *d, void *c, int len, float m1, float m2,
                          int64_t ld) {
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<16>::run(
      d, c, len, m1, m2);
}
