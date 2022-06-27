#pragma once
#include <immintrin.h>

static inline __m512 _mm512_mlperf_erf_ps(__m512 x) {
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
  auto y = _mm512_mlperf_erf_ps(x * rsqrt_2) + _mm512_set1_ps(1);

  return x * _mm512_set1_ps(0.5f) * y;
}

static inline __m512i _mm512_scale_minmax_gelu_i8_ps(__m512 x, __m512 vS, __m512 vS2) {
  auto max = _mm512_set1_ps(127.f);
  auto min = _mm512_set1_ps(-127.f);

  auto r = x * vS;
  auto g = _mm512_gelu_ps(r);
  auto m = _mm512_roundscale_ps(g * vS2, _MM_FROUND_TO_NEAREST_INT);

  auto c1 = _mm512_min_ps(m, max);
  auto c2 = _mm512_max_ps(c1, min);
  return _mm512_cvtps_epi32(c2);
}

static inline __m512i _mm512_scale_minmax_i8_ps(__m512 x, __m512 vS) {
  auto max = _mm512_set1_ps(127.f);
  auto min = _mm512_set1_ps(-127.f);

  auto m = _mm512_roundscale_ps(x * vS, _MM_FROUND_TO_NEAREST_INT);
  auto c1 = _mm512_min_ps(m, max);
  auto c2 = _mm512_max_ps(c1, min);
  return _mm512_cvtps_epi32(c2);
}

static inline __m512i _mm512_scale_minmax_i8_ph(__m512h x, __m512h vS) {
  auto max = _mm512_set1_ph(127.f);
  auto min = _mm512_set1_ph(-127.f);

  auto m = _mm512_roundscale_ph(x * vS, _MM_FROUND_TO_NEAREST_INT);
  auto c1 = _mm512_min_ph(m, max);
  auto c2 = _mm512_max_ph(c1, min);
  return _mm512_cvtph_epi16(c2);
}

static inline void _mm512_mask_cvtepi32_storeu_epi8_compensate(void *base_addr,
                                                               __mmask16 k,
                                                               __m512i x,
                                                               __m128i off) {
  auto z = _mm512_cvtepi32_epi8(x);
  auto o = z ^ off;
  _mm_mask_storeu_epi8(base_addr, k, o);
}

static inline __m256 _mm256_max_reduce_ps(__m256 v) {
  auto perm0 = _mm256_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
  auto m1 = _mm256_max_ps(v, perm0);
  auto perm1 = _mm256_permute_ps(m1, _MM_SHUFFLE(1, 0, 3, 2));
  auto m2 = _mm256_max_ps(perm1, m1);
  auto perm2 = _mm256_permute2f128_ps(m2, m2, 0x01);
  auto m3 = _mm256_max_ps(perm2, m2);
  return m3;
}

static inline float _mm256_reduce_max_ps(__m256 v) {
  return _mm256_max_reduce_ps(v)[0];
}

static inline __m256 _mm256_add_reduce_ps(__m256 v) {
  auto perm0 = _mm256_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
  auto m1 = v + perm0;
  auto perm1 = _mm256_permute_ps(m1, _MM_SHUFFLE(1, 0, 3, 2));
  auto m2 = m1 + perm1;
  auto perm2 = _mm256_permute2f128_ps(m2, m2, 0x01);
  auto m3 = m2 + perm2;
  return m3;
}

static inline float _mm256_reduce_add_ps(__m256 v) {
  return _mm256_add_reduce_ps(v)[0];
}

static inline __m512 _mm512_max_reduce_ps(__m512 v) {
  auto perm0 = _mm512_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
  auto m1 = _mm512_max_ps(v, perm0);
  auto perm1 = _mm512_permute_ps(m1, _MM_SHUFFLE(1, 0, 3, 2));
  auto m2 = _mm512_max_ps(perm1, m1);
  auto perm2 = _mm512_shuffle_f32x4(m2, m2, _MM_SHUFFLE(2, 3, 0, 1));
  auto m3 = _mm512_max_ps(perm2, m2);
  auto perm3 = _mm512_shuffle_f32x4(m3, m3, _MM_SHUFFLE(1, 0, 3, 2));
  auto m4 = _mm512_max_ps(perm3, m3);
  return m4;
}

static inline __m512 _mm512_add_reduce_ps(__m512 v) {
  auto perm0 = _mm512_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
  auto m1 = v + perm0;
  auto perm1 = _mm512_permute_ps(m1, _MM_SHUFFLE(1, 0, 3, 2));
  auto m2 = m1 + perm1;
  auto perm2 = _mm512_shuffle_f32x4(m2, m2, _MM_SHUFFLE(2, 3, 0, 1));
  auto m3 = m2 + perm2;
  auto perm3 = _mm512_shuffle_f32x4(m3, m3, _MM_SHUFFLE(1, 0, 3, 2));
  auto m4 = m3 + perm3;
  return m4;
}

static inline __m512h _mm512_half_add_reduce_ph(__m512h v) {
  /*
  do add reduce each half separately
  */
  // 1 round shuffle -> 16 bits
  auto perm_idx = _mm512_set_epi16(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 
                                   17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
  auto perm0 = _mm512_permutexvar_ph(perm_idx, v);
  auto m1 = v + perm0;
  auto m1ps = _mm512_castph_ps(m1);
  // 2 round shuffle -> 32 bits
  auto perm1 = _mm512_permute_ps(m1ps, _MM_SHUFFLE(2, 3, 0, 1));
  auto m2 = m1ps + perm1;

  auto perm2 = _mm512_permute_ps(m2, _MM_SHUFFLE(1, 0, 3, 2));
  auto m3 = m2 + perm2;

  auto perm3 = _mm512_shuffle_f32x4(m3, m3, _MM_SHUFFLE(1, 0, 3, 2));
  auto m4 = m3 + perm3;

  return _mm512_castps_ph(m4);
}

static inline void _mm512_cvtepi16_epi8_shuffle_storeu(void* mem_addr, int ld, __m512i i0, __m512i i1, __m512i i2, __m512i i3) {
  // shuffle again to store
  auto i0_hor = _mm512_shuffle_i32x4(i0, i1, _MM_SHUFFLE(0, 1, 0, 1));
  auto i1_hor = _mm512_shuffle_i32x4(i0, i1, _MM_SHUFFLE(2, 3, 2, 3));
  auto i2_hor = _mm512_shuffle_i32x4(i2, i3, _MM_SHUFFLE(0, 1, 0, 1));
  auto i3_hor = _mm512_shuffle_i32x4(i2, i3, _MM_SHUFFLE(2, 3, 2, 3));

  auto pout = reinterpret_cast<int8_t(*)[ld]>(mem_addr);
  _mm512_mask_cvtepi16_storeu_epi8(pout[0], 0xffff, i0_hor);
  _mm512_mask_cvtepi16_storeu_epi8(pout[0] + 32, 0xffff, i2_hor);
  _mm512_mask_cvtepi16_storeu_epi8(pout[1], 0xffff, i1_hor);
  _mm512_mask_cvtepi16_storeu_epi8(pout[1] + 32, 0xffff, i3_hor);
}

inline static __m512 _mm512_loadu_i8_to_fp32(void const *mem_addr) {
  auto l = _mm_loadu_si128((__m128i *)mem_addr);
  auto i = _mm512_cvtepi8_epi32(l);
  return _mm512_cvtepi32_ps(i);
}

inline static __m512 _mm512_loadu_c8_to_fp32(void const *mem_addr) {
  auto l = _mm_loadu_si128((__m128i *)mem_addr);
  auto c = _mm_set1_epi32(0x80808080);
  auto decomp = _mm_xor_si128(c, l);
  auto i = _mm512_cvtepi8_epi32(decomp);
  return _mm512_cvtepi32_ps(i);
}

inline static __m512 _mm512_mask_loadu_i8_to_fp32(__m128i src, __mmask64 k,
                                                  void const *mem_addr) {
  auto l = _mm_mask_loadu_epi8(src, k, mem_addr);
  auto i = _mm512_cvtepi8_epi32(l);
  return _mm512_cvtepi32_ps(i);
}

inline static __m512 _mm512_mask_loadu_c8_to_fp32(__m128i src, __mmask64 k,
                                                  void const *mem_addr) {
  auto l = _mm_mask_loadu_epi8(src, k, mem_addr);
  auto c = _mm_set1_epi32(0x80808080);
  auto decomp = _mm_xor_si128(c, l);
  auto i = _mm512_cvtepi8_epi32(decomp);
  return _mm512_cvtepi32_ps(i);
}

inline static __m512 _mm512_loadu_i32_to_fp32(void const *mem_addr) {
  auto l = _mm512_loadu_si512(mem_addr);
  return _mm512_cvtepi32_ps(l);
}

inline static __m512 _mm512_mask_loadu_i32_to_fp32(__m512i src, __mmask64 k,
                                                   void const *mem_addr) {
  auto l = _mm512_mask_load_epi32(src, k, mem_addr);
  return _mm512_cvtepi32_ps(l);
}

inline static __m512 _mm512_mean_reduce_ps(__m512 v, int64_t N) {
  auto rN = _mm512_set1_ps(1. / N);
  auto vsum = _mm512_add_reduce_ps(v);
  return vsum * rN;
}

static inline __m256 snd_order_poly_exp(__m256 z, __m256 f, const float c[]) {
  const auto c0 = _mm256_set1_ps(c[0]);
  const auto c1 = _mm256_set1_ps(c[1]);
  const auto c2 = _mm256_set1_ps(c[2]);

  auto y = (f * c0 + c1) * f + c2;
  auto exp = _mm256_scalef_ps(y, z);
  return exp;
}

static inline __m512 snd_order_poly_exp(__m512 z, __m512 f, const float c[]) {
  const auto c0 = _mm512_set1_ps(c[0]);
  const auto c1 = _mm512_set1_ps(c[1]);
  const auto c2 = _mm512_set1_ps(c[2]);

  auto y = (f * c0 + c1) * f + c2;
  auto exp = _mm512_scalef_ps(y, z);

  return exp;
}

static inline __m512h snd_order_poly_exp_ph(__m512h z, __m512h f, const _Float16 c[]) {
  const auto c0 = _mm512_set1_ph(c[0]);
  const auto c1 = _mm512_set1_ph(c[1]);
  const auto c2 = _mm512_set1_ph(c[2]);

  auto y = (f * c0 + c1) * f + c2;
  auto exp = _mm512_scalef_ph(y, z);

  return exp;
}

static inline __m256 third_order_poly_exp(__m256 z, __m256 f, const float c[]) {
  const auto c0 = _mm256_set1_ps(c[0]);
  const auto c1 = _mm256_set1_ps(c[1]);
  const auto c2 = _mm256_set1_ps(c[2]);
  const auto c3 = _mm256_set1_ps(c[3]);

  auto y = ((f * c0 + c1) * f + c2) * f + c3;
  auto exp = _mm256_scalef_ps(y, z);

  return exp;
}

static inline __m512 third_order_poly_exp(__m512 z, __m512 f, const float c[]) {
  const auto c0 = _mm512_set1_ps(c[0]);
  const auto c1 = _mm512_set1_ps(c[1]);
  const auto c2 = _mm512_set1_ps(c[2]);
  const auto c3 = _mm512_set1_ps(c[3]);

  auto y = ((f * c0 + c1) * f + c2) * f + c3;
  auto exp = _mm512_scalef_ps(y, z);

  return exp;
}

// [0.5, 0.5)
static inline __m256 exp_ps_0_1(__m256 x) {
  const auto log2e = _mm256_set1_ps(1.442695f);
  const float _c[] = {0.240226507f, 0.452920674f, 0.713483036f};

  auto x1 = x * log2e + _mm256_set1_ps(0.5f);
  auto z = _mm256_floor_ps(x1);
  auto f = x1 - z;

  return snd_order_poly_exp(z, f, _c);
}

// [0.5, 0.5)
static inline __m512 exp_ps_0_1(__m512 x) {
  const auto log2e = _mm512_set1_ps(1.442695f);
  const float _c[] = {0.240226507f, 0.452920674f, 0.713483036f};

  auto x1 = x * log2e + _mm512_set1_ps(0.5f);
  auto z = _mm512_floor_ps(x1);
  auto f = x1 - z;

  return snd_order_poly_exp(z, f, _c);
}

// [0.5, 0.5)
static inline __m512h exp_ph_0_1(__m512h x) {
  const auto log2e = _mm512_set1_ph(1.442695f);
  const _Float16 _c[] = {0.240226507f, 0.452920674f, 0.713483036f};

  auto x1 = x * log2e + _mm512_set1_ph(0.5f);
  auto z = _mm512_cvtepi16_ph(_mm512_cvt_roundph_epi16(x1, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC));
  auto f = x1 - z;

  return snd_order_poly_exp_ph(z, f, _c);
}

static inline __m256 exp_ps_zero_one_third(__m256 x) {
  const auto log2e = _mm256_set1_ps(1.442695f);
  const auto half = _mm256_set1_ps(0.5f);
  const float _c[] = {0.05550410866f, 0.15697034396f, 0.49454875509f,
                      0.70654502287f};

  auto x1 = x * log2e + half;
  auto z = _mm256_floor_ps(x1);
  auto f = x1 - z;

  return third_order_poly_exp(z, f, _c);
}

static inline __m512 exp_ps_zero_one_third(__m512 x) {
  const auto log2e = _mm512_set1_ps(1.442695f);
  const auto half = _mm512_set1_ps(0.5f);
  const float _c[] = {0.05550410866f, 0.15697034396f, 0.49454875509f,
                      0.70654502287f};

  auto x1 = x * log2e + half;
  auto z = _mm512_floor_ps(x1);
  auto f = x1 - z;

  return third_order_poly_exp(z, f, _c);
}

// Smaller range [-ln2, 0)
static inline __m256 exp_ps_negln2_zero(__m256 x) {
  const auto _log2e = _mm256_set1_ps(1.442695f);
  const auto ln2 = _mm256_set1_ps(0.693147180f);
  const float _c[] = {0.35815147f, 0.96963238f, 1.0f};

  auto z = _mm256_ceil_ps(x * _log2e);
  auto f = x - z * ln2;

  return snd_order_poly_exp(z, f, _c);
}

static inline __m512 exp_ps_negln2_zero(__m512 x) {
  const auto _log2e = _mm512_set1_ps(1.442695f);
  const auto ln2 = _mm512_set1_ps(0.693147180f);
  const float _c[] = {0.35815147f, 0.96963238f, 1.0f};

  auto z = _mm512_ceil_ps(x * _log2e);
  auto f = x - z * ln2;

  return snd_order_poly_exp(z, f, _c);
}
