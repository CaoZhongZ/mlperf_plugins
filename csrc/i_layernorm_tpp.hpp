#pragma once
#include <immintrin.h>

namespace intel_mlperf {
template <int vec_length>
class i_residual_layernorm_tpp {
public:
  static void ref(
      int8_t *out, int8_t *src1, int8_t *src2, float *weight, float *bias,
      float s1, float s2, float oscale, int64_t rl, float eps=1e-05);

  static void ref(
      int8_t *out, int8_t *src1, int8_t *src2, int8_t * src3,
      float *weight, float *bias, float s1, float s2, float oscale,
      int64_t rl, float eps=1e-05);
};

template <int vec_length>
class i_layernorm_tpp {
public:
  static void ref(
      int8_t *out, float *in, float *weight, float *bias,
      float oscale, int64_t rl, float eps=1e-05);

  static void ref(
      int8_t *out, int8_t *in, float *weight, float *bias,
      float oscale, int64_t rl, float eps=1e-05);
};
}
