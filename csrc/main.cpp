#include <cstring>
#include <cstdlib>
#include "transpose.hpp"
#include "helper.hpp"

#include "i_softmax_tpp.hpp"

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

  intel_mlperf::tr_vnni_4x<4>((void*)b, (void*)a, lda, 64);
  intel_mlperf::print_2d_matrix<int8_t>((int8_t*)b, 4, 64, 64);

  int32_t c[32][384];
  int8_t d[32][384];
  
  memset(c, 5, sizeof(b));
  memset(d, 0, sizeof(d));

  softmax_isolation((void*)d, (void*)c, 384, 1.0, 1.0, 384);

  return 0;
}

void softmax_isolation(void *d, void *c, int len, float m1, float m2, int64_t ld) {
  intel_mlperf::i32_scale_attlen_softmax_scale_i8<16, 8>::run(d, c, len, m1, m2, ld);
}
