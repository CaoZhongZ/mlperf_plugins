#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include "transpose.hpp"
#include "helper.hpp"
#include "i_softmax_tpp.hpp"

#include "i_softmax_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

void softmax_isolation(void *d, void *c, int len, float m1, float m2, int64_t ld);

int main() {
  size_t a_row = 4;
  size_t lda = 64;

  int8_t a[a_row][lda];

  for (size_t j = 0; j < a_row; ++j) {
    for (size_t i = 0; i < lda; ++i) {
      a[j][i] = i;
    }
  }

  intel_mlperf::print_2d_matrix<int8_t>((int8_t*)a, 4, 64, lda);
  getchar();

  int8_t b[4][64];
  memset(b, 0, sizeof(b));

  // intel_mlperf::tr_vnni_4x<4>((void*)b, (void*)a, lda, 64);
  // intel_mlperf::print_2d_matrix<int8_t>((int8_t*)b, 4, 64, 64);

  int32_t c[32][384];
  int8_t d[32][384];
  
  memset(c, 5, sizeof(b));
  memset(d, 0, sizeof(d));

  softmax_isolation((void*)d, (void*)c, 384, 1.0, 1.0, 384);

  auto start = Time::now();

  // for (int i = 0; i < 100000; i++)
    intel_mlperf::i32_scale_attlen_softmax_scale_i8<16, 8>::run((void*)d, (void*)c, 384, 1.0, 1.0, 384);

  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  std::cout << "100000 times softmax time : " << (float)during / 1000 / 1000 << " ms " << std::endl;

  start = Time::now();
  // for (int i = 0; i < 100000; i++)
    softmax_isolation((void*)d, (void*)c, 384, 1.0, 1.0, 384);

  during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  std::cout << "100000 times tile softmax time : " << (float)during / 1000 / 1000 << " ms " << std::endl;

  return 0;
}

void softmax_isolation(void *d, void *c, int len, float m1, float m2, int64_t ld) {
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni::run(d, c, len, m1, m2);
}
void softmax_isolation_16(void *d, void *c, int len, float m1, float m2, int64_t ld) {
  intel_mlperf::i32_scale_attlen_softmax_scale_i8_amx_tile_vnni::run(d, c, len, m1, m2);
}
