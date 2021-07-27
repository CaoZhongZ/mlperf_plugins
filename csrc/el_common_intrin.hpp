#pragma once
#include <immintrin.h>

static inline __m512i _mm512_scale_minmax_i8_ps(__m512 x, __m512 vS) {
  auto max = _mm512_set1_ps(127);
  auto min = _mm512_set1_ps(-127);

  auto m = _mm512_roundscale_ps(x * vS, _MM_FROUND_TO_NEAREST_INT);
  auto c1 = _mm512_min_ps(m, max);
  auto c2 = _mm512_max_ps(c1, min);
  return _mm512_cvtps_epi32(c2);
}

static inline __m256 _mm256_max_reduce_ps(__m256 v) {
  auto perm0 = _mm256_permute_ps(v, _MM_SHUFFLE(2,3,0,1));
  auto m1 = _mm256_max_ps(v, perm0);
  auto perm1 = _mm256_permute_ps(m1, _MM_SHUFFLE(1,0,3,2));
  auto m2 = _mm256_max_ps(perm1, m1);
  auto perm2 = _mm256_permute2f128_ps(m2, m2, 0x01);
  auto m3 = _mm256_max_ps(perm2, m2);
  return m3;
}

static inline float _mm256_reduce_max_ps(__m256 v) {
  return _mm256_max_reduce_ps(v)[0];
}

static inline __m256 _mm256_add_reduce_ps(__m256 v) {
  auto perm0 = _mm256_permute_ps(v, _MM_SHUFFLE(2,3,0,1));
  auto m1 = v + perm0;
  auto perm1 = _mm256_permute_ps(m1, _MM_SHUFFLE(1,0,3,2));
  auto m2 = m1 + perm1;
  auto perm2 = _mm256_permute2f128_ps(m2, m2, 0x01);
  auto m3 = m2 + perm2;
  return m3;
}

static inline float _mm256_reduce_add_ps(__m256 v) {
  return _mm256_add_reduce_ps(v)[0];
}

static inline __m512 _mm512_max_reduce_ps(__m512 v) {
  auto perm0 = _mm512_permute_ps(v, _MM_SHUFFLE(2,3,0,1));
  auto m1 = _mm512_max_ps(v, perm0);
  auto perm1 = _mm512_permute_ps(m1, _MM_SHUFFLE(1,0,3,2));
  auto m2 = _mm512_max_ps(perm1, m1);
  auto perm2 = _mm512_shuffle_f32x4(m2, m2, _MM_SHUFFLE(2,3,0,1));
  auto m3 = _mm512_max_ps(perm2, m2);
  auto perm3 = _mm512_shuffle_f32x4(m3, m3, _MM_SHUFFLE(1,0,3,2));
  auto m4 = _mm512_max_ps(perm3, m3);
  return m4;
}

static inline __m512 _mm512_add_reduce_ps(__m512 v) {
  auto perm0 = _mm512_permute_ps(v, _MM_SHUFFLE(2,3,0,1));
  auto m1 = v + perm0;
  auto perm1 = _mm512_permute_ps(m1, _MM_SHUFFLE(1,0,3,2));
  auto m2 = m1 + perm1;
  auto perm2 = _mm512_shuffle_f32x4(m2, m2, _MM_SHUFFLE(2,3,0,1));
  auto m3 = m2 + perm2;
  auto perm3 = _mm512_shuffle_f32x4(m3, m3, _MM_SHUFFLE(1,0,3,2));
  auto m4 = m3 + perm3;
  return m4;
}
