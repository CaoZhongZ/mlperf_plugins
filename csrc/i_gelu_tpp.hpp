#pragma once
#include <immintrin.h>

namespace intel_mlperf {
template <int vec_length>
class i_gelu_tpp {
public:
  //
  // M is input integer tensor scaling factor
  // 
  static void ref(
      void *out, void *in, float M, float oscale, int64_t nelem);
};
}
