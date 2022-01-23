#pragma once

#include <immintrin.h>

namespace intel_mlperf {

// Transpose N*16 integer shape into padded area
// avx512 above
//
// Single step of transpose 16*16
template <int tail>
inline void tr_x16(int *at, const int *a, size_t lda, size_t ldat) {
  auto a_ = reinterpret_cast<int (*)[lda]>(a);

  __m512i even[16];

# pragma unroll (tail)
  for (int i = 0; i < tail; ++ i) {
    even[i] = _mm512_loadu_epi32(a_ + i);
  }

# pragma unroll (16 - tail)
  for (int i = tail; i < 16; ++i) {
    even[i] = _mm512_set1_epi32(0);
  }

  __m512i odd[16];

# pragma unroll (8)
  for (int i = 0; i < 8; ++ i) {
    odd[2i] = _mm512_unpacklo_epi32(even[2i], even[2i+1]);
    odd[2i + 1] = _mm512_unpackhi_epi32(even[2i], even[2i+1]);
  }

# pragma unroll (4)
  for (int i = 0; i < 4; ++ i) {
    even[4i] = _mm512_unpacklo_epi64(odd[4i], odd[4i + 2]);
    even[4i + 1] = _mm512_unpackhi_epi64(odd[4i], odd[4i + 2]);
    even[4i + 2] = _mm512_unpacklo_epi64(odd[4i + 1], odd[4i + 3]);
    even[4i + 3] = _mm512_unpackhi_epi64(odd[4i + 1], odd[4i + 3]);
  }

# pragma unroll (2)
  for (int i = 0; i < 2; ++ i ){
    odd[8i+0] = _mm512_shuffle_i32x4(even[8i+0], even[8i + 4], 0x88);
    odd[8i+1] = _mm512_shuffle_i32x4(even[8i+1], even[8i + 5], 0x88);
    odd[8i+2] = _mm512_shuffle_i32x4(even[8i+2], even[8i + 6], 0x88);
    odd[8i+3] = _mm512_shuffle_i32x4(even[8i+3], even[8i + 7], 0x88);
    odd[8i+4] = _mm512_shuffle_i32x4(even[8i+0], even[8i + 4], 0xdd);
    odd[8i+5] = _mm512_shuffle_i32x4(even[8i+1], even[8i + 5], 0xdd);
    odd[8i+6] = _mm512_shuffle_i32x4(even[8i+2], even[8i + 6], 0xdd);
    odd[8i+7] = _mm512_shuffle_i32x4(even[8i+3], even[8i + 7], 0xdd);
  }

  even[0] = _mm512_shuffle_i32x4(odd[0], odd[8], 0x88);
  even[1] = _mm512_shuffle_i32x4(odd[1], odd[9], 0x88);
  even[2] = _mm512_shuffle_i32x4(odd[2], odd[10], 0x88);
  even[3] = _mm512_shuffle_i32x4(odd[3], odd[11], 0x88);
  even[4] = _mm512_shuffle_i32x4(odd[4], odd[12], 0x88);
  even[5] = _mm512_shuffle_i32x4(odd[5], odd[13], 0x88);
  even[6] = _mm512_shuffle_i32x4(odd[6], odd[14], 0x88);
  even[7] = _mm512_shuffle_i32x4(odd[7], odd[15], 0x88);
  even[8] = _mm512_shuffle_i32x4(odd[0], odd[8], 0xdd);
  even[9] = _mm512_shuffle_i32x4(odd[1], odd[9], 0xdd);
  even[10] = _mm512_shuffle_i32x4(odd[2], odd[10], 0xdd);
  even[11] = _mm512_shuffle_i32x4(odd[3], odd[11], 0xdd);
  even[12] = _mm512_shuffle_i32x4(odd[4], odd[12], 0xdd);
  even[13] = _mm512_shuffle_i32x4(odd[5], odd[13], 0xdd);
  even[14] = _mm512_shuffle_i32x4(odd[6], odd[14], 0xdd);
  even[15] = _mm512_shuffle_i32x4(odd[7], odd[15], 0xdd);

  auto at_ = reinterpret_cast<int (*)[ldat]>(at);

# pragma unroll (16)
  for (int i = 0; i < 16; ++ i) {
    _mm512_storeu_epi32(at_ + i, even[i]);
  }
}

//
// Transpose 4*64 to 64*4
//
template <int tail, int group=1>
inline void i8_tr_4x(void *out, void *a, size_t lda, size_t ldo) {
  __m512i row[4];
  auto a_ = reinterpret_cast<int (*)[lda]>(a);

# pragma unroll (tail)
  for (int i = 0; i < tail; ++i)
    row[i] = _mm512_loadu_epi8(a + i);

# pragma unroll (4 - tail)
  for (int i = tail; i < 4; ++i)
    row[i] = _mm512_set1_epi8(0);

  auto __t0 = _mm512_unpacklo_epi8(row[0], row[1]);
  auto __t1 = _mm512_unpackhi_epi8(row[0], row[1]);
  auto __t2 = _mm512_unpacklo_epi8(row[2], row[3]);
  auto __t3 = _mm512_unpackhi_epi8(row[2], row[3]);

  auto _tt0 = _mm512_unpacklo_epi16(__t0, __t2);
  auto _tt1 = _mm512_unpackhi_epi16(__t0, __t2);
  auto _tt2 = _mm512_unpacklo_epi16(__t1, __t3);
  auto _tt3 = _mm512_unpackhi_epi16(__t1, __t3);

  auto new0 = _mm512_shuffle_i32x4(_tt0, _tt1, 0x88);
  auto new1 = _mm512_shuffle_i32x4(_tt2, _tt3, 0x88);
  auto new2 = _mm512_shuffle_i32x4(_tt0, _tt1, 0xdd);
  auto new3 = _mm512_shuffle_i32x4(_tt2, _tt3, 0xdd);

  auto nn0 = _mm512_shuffle_i32x4(new0, new1, 0x88);
  auto nn1 = _mm512_shuffle_i32x4(new0, new1, 0xdd);
  auto nn3 = _mm512_shuffle_i32x4(new2, new3, 0x88);
  auto nn4 = _mm512_shuffle_i32x4(new2, new3, 0xdd);

  auto o_ = reinterpret_cast<int (*)[ldo]>(out);

  if ( group == 1 ) {
    _mm512_store_epi32(o_+0, nn0);
    _mm512_store_epi32(o_+1, nn1);
    _mm512_store_epi32(o_+2, nn2);
    _mm512_store_epi32(o_+3, nn3);
  } else (group == 2) {
    // TODO: 128 * 2 group
  }
}

}
