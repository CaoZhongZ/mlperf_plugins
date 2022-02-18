#pragma once

#include <cstdlib>
#include <immintrin.h>
#include "amx_tdpbssd.hpp"

namespace intel_mlperf {

template <int row_tile, int col_tile>
struct _tile_dot_product_16x256 {

  static constexpr size_t A_footprint = row_tile * 16 * 1024;
  static constexpr size_t B_footprint = col_tile * 16 * 256;
  static constexpr size_t cache_footprint = A_footprint + B_footprint;

  inline static void load_A(void *A) {
    auto A_ = reinterpret_cast<int8_t *>(A);
    const size_t A_block = 16 * col_tile * 64;

    _tile_loadd(TMM4, A_, 64);
    if (row_tile == 2) _tile_loadd(TMM5, A_ + A_block, 64);
  }

  inline static void load_B(void *B) {
    auto B_ = reinterpret_cast<int8_t (*)[16][64]>(B);
    _tile_loadd(TMM6, B_[0], 64);
    _tile_loadd(TMM7, B_[1], 64);
  }

  inline static void dot_prod() {
    __tile_dpbssd<TMM0, TMM4, TMM6>();
    if (row_tile == 2)
      __tile_dpbssd<TMM2, TMM5, TMM6>();
    __tile_dpbssd<TMM1, TMM4, TMM7>();
    if (row_tile == 2)
      __tile_dpbssd<TMM3, TMM5, TMM7>();
  }

  inline static void store(void *S) {
    auto S_ = reinterpret_cast<int (*)[16][16]>(S);
    _tile_stored(TMM0, S_[0], 64);
    _tile_stored(TMM1, S_[1], 64);
    if (row_tile == 2) {
      _tile_stored(TMM2, S_[2], 64);
      _tile_stored(TMM3, S_[3], 64);
    }
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
  }

  inline static void quant_out(void *C, void *s_0, void *s_1, float *bias, float scale, size_t rollback) {
    auto s_0_ = reinterpret_cast<int (*)[2][16][16]>(s_0);
    auto s_1_ = reinterpret_cast<int (*)[2][16][16]>(s_1);

    auto scale_ = _mm512_set1_ps(scale);
    constexpr size_t c_block = 16 * col_tile * 64;
    auto bias_ = reinterpret_cast<float (*)[16]>(bias);

    auto b0 = _mm512_loadu_ps(bias_[0]);
    auto b1 = _mm512_loadu_ps(bias_[1]);
    auto b2 = _mm512_loadu_ps(bias_[2]);
    auto b3 = _mm512_loadu_ps(bias_[3]);

#   pragma unroll (row_tile)
    for (int t = 0; t < row_tile; ++ t) {
      auto C_ = reinterpret_cast<int8_t (*)[4][16]>((int8_t *)C + c_block * t);

#     pragma unroll (16)
      for (int i = 0; i < 16; ++ i) {
        if (t == row_tile - 1 && i > 16 - rollback)
          break;

        auto i0 = _mm512_load_epi32(s_0[t][0][i]);
        auto i1 = _mm512_load_epi32(s_0[t][1][i]);
        auto i2 = _mm512_load_epi32(s_1[t][0][i]);
        auto i3 = _mm512_load_epi32(s_1[t][1][i]);

        auto f0 = _mm512_cvtepi32_ps(i0) + b0;
        auto f1 = _mm512_cvtepi32_ps(i1) + b1;
        auto f2 = _mm512_cvtepi32_ps(i2) + b2;
        auto f3 = _mm512_cvtepi32_ps(i3) + b3;

        auto o0 = _m512_scale_minmax_i8_ps(scale_, f0);
        auto o1 = _m512_scale_minmax_i8_ps(scale_, f1);
        auto o2 = _m512_scale_minmax_i8_ps(scale_, f2);
        auto o3 = _m512_scale_minmax_i8_ps(scale_, f3);

        _mm512_mask_cvtepi32_storeu_epi8(C_[i][0], o0);
        _mm512_mask_cvtepi32_storeu_epi8(C_[i][1], o1);
        _mm512_mask_cvtepi32_storeu_epi8(C_[i][2], o2);
        _mm512_mask_cvtepi32_storeu_epi8(C_[i][3], o3);
      }
    }
  }

  // Pure tile format
  inline static void compute(void *C, void *A, void *B, float *bias, float scale, size_t rollback = 0) {
    alignas (64) int scratch_0[row_tile][2][16][16];
    alignas (64) int scratch_1[row_tile][2][16][16];

    auto A_ = reinterpret_cast<int8_t (*)[16][64]>(A);
    auto B_ = reinterpret_cast<int8_t (*)[32][64]>(B);

    zero_accum();

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      load_A(A_[i]);
      load_B(B_[i]);
      dot_prod();
    }

    store(scratch_0);

    zero_accum();
    B_ = reinterpret_cast<int8_t (*)[32][64]>((int8_t *)B + 256 * 128);

#   pragma unroll (col_tile)
    for (int i = 0; i < col_tile; ++i) {
      load_A(A_[i]);
      load_B(B_[i]);
      dot_prod();
    }

    store(scratch_1);
    quant_out(C, scratch_0, scratch_1, bias, scale, rollback);
  }
};

}
