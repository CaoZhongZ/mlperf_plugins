#include "i_gelu_tpp.hpp"
#include "el_common_intrin.hpp"

namespace intel_mlperf {

static inline __m512 _mm512_erf_ps(__m512 x) {
  auto a = _mm512_set1_ps(-0.2888f);
  auto b = _mm512_set1_ps(1.0217744f);
  auto c = _mm512_set1_ps(0.0962405432f);

  auto nb = _mm512_set1_ps(1.769f);
  auto m = _mm512_set1_epi32(0x80000000);

  auto ix = _mm512_castps_si512(x);
  auto s = _mm512_and_epi32(m, ix);
  auto abs = _mm512_abs_ps(x);

  auto v = _mm512_min_ps(abs, nb);
  auto y = (a * v + b) * v + c;
  auto z = _mm512_or_epi32(_mm512_castps_si512(y), s);

  return _mm512_castsi512_ps(z);
}

static inline __m512 _mm512_gelu_ps(__m512 x) {
  auto rsqrt_2 = _mm512_set1_ps(0.70710678);
  auto y = _mm512_erf_ps(x * rsqrt_2) + 1;

  return x * 0.5f * y;
}

// Cover only to N [1, 16]
// User's responsibility for tail access
//
template <int vec_l, int N>
struct i32_scale_gelu_scale_i8 {
  inline static void run(
      int8_t *out, int32_t *in, float M, float oscale);
  inline static void run(
      int8_t *out, int32_t *in, float M, float oscale, __mmask16 tail);
};

template <int N>
struct i32_scale_gelu_scale_i8<16, N> {
  static constexpr int64_t batch = 16 * N;

  inline static void run(
      int8_t *out, int32_t *in, float M, float oscale) {
    auto vM = _mm512_set1_ps(M);
    auto vS = _mm512_set1_ps(oscale);

#   pragma unroll N
    for (int i = 0; i < N; ++ i) {
      auto x = _mm512_load_epi32(&in[i * 16]);
      auto f = _mm512_cvtepi32_ps(x) * vM;
      auto o = _mm512_gelu_ps(f);
      auto z = _mm512_scale_minmax_i8_ps(o, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z);
    }
  }

  inline static void run(
      int8_t *out, int32_t *in, float M, float oscale, __mmask16 tail) {
    auto vM = _mm512_set1_ps(M);
    auto vS = _mm512_set1_ps(oscale);

#   pragma unroll N -1
    for (int i = 0; i < N -1; ++ i) {
      auto x = _mm512_load_epi32(&in[i * 16]);
      auto f = _mm512_cvtepi32_ps(x) * vM;
      auto o = _mm512_gelu_ps(f);
      auto z = _mm512_scale_minmax_i8_ps(o, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], 0xffff, z);
    }
    {
      auto i = N-1;
      auto zero = _mm512_setzero_epi32();
      auto x = _mm512_mask_loadu_epi32(zero, tail, &in[i * 16]);
      auto f = _mm512_cvtepi32_ps(x) * vM;
      auto o = _mm512_gelu_ps(f);
      auto z = _mm512_scale_minmax_i8_ps(o, vS);
      _mm512_mask_cvtepi32_storeu_epi8(&out[i * 16], tail, z);
    }
  }
};

//
// expecting nelem is integer multiply of batch, expand functionality in
// the furture
//
template <int vec_length>
void i_gelu_tpp<vec_length>::ref(
    void *out, void *in, float M, float oscale, int64_t nelem) {

  auto constexpr b = i32_scale_gelu_scale_i8<vec_length, 16>::batch;
  auto n_batch = nelem / b;

  auto pin = reinterpret_cast<int32_t *>(in);
  auto pout = reinterpret_cast<int8_t *>(out);

  for (int p = 0; p < n_batch; ++ p, pout += b, pin += b) {
    i32_scale_gelu_scale_i8<vec_length, 16>::run(pout, pin, M, oscale);
  }
}

template void i_gelu_tpp<16>::ref(void *, void *, float, float, int64_t);

}
