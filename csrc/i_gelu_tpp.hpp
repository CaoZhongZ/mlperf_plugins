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

template <int vec_length>
class i_identity_tpp {
public:
  //
  // M is input integer tensor scaling factor
  //
  static void ref(
      int8_t *out, int32_t *in, float M, float oscale, int64_t nelem);
  static void ref(
      int8_t *out, float *in, float oscale, int64_t nelem);
};

}
