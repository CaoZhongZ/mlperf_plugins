#include "reflection_copy_tpp.hpp"

#include <cstdlib>
#include <stdexcept>

#include "el_common_intrin.hpp"

namespace intel_mlperf {

void reflection_copy_tpp::ref(float *pout, float *pin, int64_t rl, int64_t d) {
  auto c0 = _mm512_set1_ps(0.0);
  for (; d < rl / 16 * 16; d += 16) {
    auto a = _mm512_loadu_ps(&pin[rl - 16 - d]);
    auto p = _mm512_permute_ps(a, _MM_SHUFFLE(0, 1, 2, 3));
    auto o = _mm512_shuffle_f32x4(p, p, _MM_SHUFFLE(0, 1, 2, 3));
    _mm512_storeu_ps(&pout[d], o);
  }

  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto a = _mm512_mask_loadu_ps(c0, k << (16 - rem), &pin[rem - 16]);
    auto p = _mm512_permute_ps(a, _MM_SHUFFLE(0, 1, 2, 3));
    auto o = _mm512_shuffle_f32x4(p, p, _MM_SHUFFLE(0, 1, 2, 3));
    _mm512_mask_storeu_ps(&pout[d], k, o);
  }
}

}  // namespace intel_mlperf
