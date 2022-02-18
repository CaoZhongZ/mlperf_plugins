#include "helper.hpp"
#include "i_softmax_tpp.hpp"
#include "transpose.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdlib.h>

#include "i_softmax_tpp.hpp"
#include "i_linear_tpp.hpp"
#include "amx_config.hpp"

using Time = std::chrono::high_resolution_clock;

template <typename T> void fill_seq(T *t, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    int start = i;
    for (size_t j = 0; j < cols; ++j)
      t[i][j] = start++;
  }
}

void set_data_act(void *a, size_t n_tile)
{
  srand(1);
  auto a_ = reinterpret_cast<int8_t (*)>(a);
  size_t elenum = n_tile * 16 * 1024;
  for (int i = 0; i < elenum; i++) {
    a_[i] = (rand() % 0xff);
  }
}

void set_data_wei(void *w, void* b) {
  srand(2);
  auto w_ = reinterpret_cast<int8_t (*)>(w);
  size_t elenum = 256 * 256;
  for (int i = 0; i < elenum; i++) {
    w_[i] = (rand() % 0xff);
  }
  auto b_ = reinterpret_cast<float (*)>(b);
  for (int i = 0; i < 256; i++) {
    b_[i] = rand();
  }
}

void softmax_isolation(void *d, void *c, int len, float m1, float m2,
                       int64_t ld);
void softmax_isolation_16(void *d, void *c, int len, float m1, float m2,
                          int64_t ld);

static constexpr size_t cols = 384;
static constexpr size_t rows = 16;

int main() {
  int32_t c[24][rows][cols];
  int8_t d[24][rows][cols];

  for (int i = 0; i < 24; ++i)
    fill_seq(c[i], rows, cols);

  memset(d, 0, sizeof(d));

  // auto start = Time::now();

  // for (int i = 0; i < 1000; i++) {
  //   for (int j = 0; j < 24; ++j) {
  //     softmax_isolation((void *)d[j], (void *)c[j], cols, 0.01, 10000, cols);
  //   }
  // }

  // auto during =
  //     std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
  //         .count();
  // std::cout << "100000 times softmax time : " << (float)during / 1000 / 1000
  //           << " ms " << std::endl;

  // start = Time::now();
  // for (int i = 0; i < 1000; i++) {
  //   for (int j = 0; j < 24; ++j) {
  //     softmax_isolation_16(d[j], c[j], cols, 0.01, 10000, cols);
  //   }
  // }

  // during =
  //     std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
  //         .count();
  // std::cout << "100000 times tile softmax time : "
  //           << (float)during / 1000 / 1000 << " ms " << std::endl;

  // start = Time::now();
  // for (int i = 0; i < 1000; i++) {
  //   for (int j = 0; j < 24; ++j) {
  //     softmax_isolation(d[j], c[j], cols, 0.01, 10000, cols);
  //   }
  // }

  // during =
  //     std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
  //         .count();
  // std::cout << "100000 times tile softmax time : "
  //           << (float)during / 1000 / 1000 << " ms " << std::endl;

  alignas(64) int8_t act[32 * 1024];
  alignas(64) int8_t wei[256 * 256];
  alignas(64) float bias[256];
  alignas(64) int8_t out[32][64];
  float scale = 2;
  set_data_act((void*)act, 2);
  set_data_wei((void*)wei, (void*)bias);

  intel_mlperf::amx_init();
  intel_mlperf::Tilecfg().set_config();
  struct cfg {
    uint8_t palette;        /* byte 0 */
    uint8_t start_row;      /* byte 1 */
    char rsvd1[14];         /* bytes 2-15 */
    uint16_t tile_colsb[8]; /* bytes 16-31 */
    char rsvd2[16];         /* bytes 32-47 */
    uint8_t tile_rows[8];   /* bytes 48-55 */
    char rsvd3[8];          /* bytes 56-63 */
  } __attribute__((packed)) cfg;

  memset(&cfg, 0, sizeof(cfg));

  for (int i = 0; i < 8; i++) {
    cfg.tile_rows[i] = 16;
    cfg.tile_colsb[i] = 64;
  }
  cfg.palette = 1;
  _tile_release();
  _tile_loadconfig(&cfg);

  intel_mlperf::_tile_dot_product_16x256<2, 16>::compute((void*)out, (void*)act, (void*)wei, (float*)bias, scale, 0);

  auto lstart = Time::now();
  for (int i = 0; i < 100000; i++) {
    intel_mlperf::_tile_dot_product_16x256<2, 16>::compute((void*)out, (void*)act, (void*)wei, (float*)bias, scale, 0);
  }
  auto lduring =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - lstart)
          .count();
  std::cout << "100000 times tile linear time : "
            << (float)lduring / 1000 / 1000 << " ms " << std::endl;
  std::cout << "single linear time : " << (float)lduring / 100000 << " ns" << std::endl;

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

#if 0
void softmax_isolation_16(void *d, void *c, int len, float m1, float m2, int64_t ld) {
  auto d_ = reinterpret_cast<int8_t (*)[64]>(d);
  auto c_ = reinterpret_cast<int (*)[16]>(c);
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<8>::run(d_, c_, len, m1, m2, len);
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<8>::run(d_ + 8, c_ + 8, len, m1, m2, len);
}
#else
void softmax_isolation_16(void *d, void *c, int len, float m1, float m2,
                          int64_t ld) {
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<16>::run(
      d, c, len, m1, m2, len);
}
#endif
