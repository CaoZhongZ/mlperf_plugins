#include <cstring>
#include <cstdlib>
#include "transpose.hpp"
#include "helper.hpp"



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

  return 0;
}
