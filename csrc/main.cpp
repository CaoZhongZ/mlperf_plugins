#include <cstring>
#include <cstdlib>
#include "transpose.hpp"

int main() {
  size_t a_row = 16;
  size_t lda = 3072;

  int8_t a[a_row][lda];

  for (size_t j = 0; j < a_row; ++j) {
    for (size_t i = 0; i < lda; ++i) {
      a[j][i] = (j * lda + i) % 127;
    }
  }

  int8_t b[16][64];
  memset(b, 0, sizeof(b));

  intel_mlperf::tr_vnni_x64<16>((void *)b, (void *)a, lda, 64);
  memset(b, 0, sizeof(b));

  intel_mlperf::tr_vnni_x64<15>((void *)b, (void *)a, lda, 64);
}
