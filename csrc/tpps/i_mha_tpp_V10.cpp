#include <assert.h>
#include <cstdlib>

#include "amx_tdpbssd.hpp"
#include "el_common_intrin.hpp"
#include "i_mha_tpp.hpp"
#include "i_softmax_tpp.hpp"
#include "transpose.hpp"

#include "helper.hpp"

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

using Time = std::chrono::high_resolution_clock;
namespace intel_mlperf {

static constexpr int max_tile_row = 16;
static constexpr int max_tile_colsb = 64;

class Tilecfg {
public:
  void set_config(bool reconfig) const {
    if (reconfig) {
      _tile_release();
      _tile_loadconfig(&cfg);
    }
  }

  void set_config() const {
    _tile_release();
    _tile_loadconfig(&cfg);
  }

  Tilecfg() {
    memset(&cfg, 0, sizeof(cfg));

    for (int i = 0; i < num_valid; i++) {
      cfg.tile_rows[i] = 16;
      cfg.tile_colsb[i] = 64;
    }
    cfg.palette = 1;
  }

  Tilecfg(int k) {
    memset(&cfg, 0, sizeof(cfg));

    // static allocation
    // A: 16 x k
    cfg.tile_rows[4] = 16;
    cfg.tile_colsb[4] = k * sizeof(int);
    cfg.tile_rows[5] = 16;
    cfg.tile_colsb[5] = k * sizeof(int);
    // B: k x 16
    cfg.tile_rows[6] = k;
    cfg.tile_colsb[6] = 16 * sizeof(int);
    cfg.tile_rows[7] = k;
    cfg.tile_colsb[7] = 16 * sizeof(int);
    // C: 16 x 16
    for (int i = 0; i < 4; i++) {
      cfg.tile_rows[i] = 16;
      cfg.tile_colsb[i] = 16 * sizeof(int);
    }
    cfg.palette = 1;
  }

private:
  static constexpr int num_valid = 8;
  struct cfg {
    uint8_t palette;        /* byte 0 */
    uint8_t start_row;      /* byte 1 */
    char rsvd1[14];         /* bytes 2-15 */
    uint16_t tile_colsb[8]; /* bytes 16-31 */
    char rsvd2[16];         /* bytes 32-47 */
    uint8_t tile_rows[8];   /* bytes 48-55 */
    char rsvd3[8];          /* bytes 56-63 */
  } __attribute__((packed)) cfg;
};

static void reorder_k_to_buffer(int8_t *k_buffer, const int8_t *k_ptr, int row,
                                int col_tile, int stride) {
  /// k_buffer format : int8_t [col_tile][16][64]
  auto k_ptr_ = reinterpret_cast<const int8_t(*)[stride]>(k_ptr);
  auto k_buffer_ = reinterpret_cast<int8_t(*)[1024]>(k_buffer);

  int i = 0;
  for (; i < col_tile - 1; i++) {
    tr_vnni_x64<16>(k_buffer_[i], k_ptr_[i * 16], stride, 64);
  }

  decltype(tr_vnni_x64<1>)* tr_vnni_tbl [] = {
    tr_vnni_x64<1>, tr_vnni_x64<2>, tr_vnni_x64<3>, tr_vnni_x64<4>,
    tr_vnni_x64<5>, tr_vnni_x64<6>, tr_vnni_x64<7>, tr_vnni_x64<8>,
    tr_vnni_x64<9>, tr_vnni_x64<10>, tr_vnni_x64<11>, tr_vnni_x64<12>,
    tr_vnni_x64<13>, tr_vnni_x64<14>, tr_vnni_x64<15>, tr_vnni_x64<16>,
  };

  int k_tail = row - (col_tile - 1) * 16;
  tr_vnni_tbl[k_tail - 1](k_buffer_[i], k_ptr_[i * 16], stride, 64);
}

static void reorder_v_to_buffer(int8_t *v_buffer, const int8_t *v_ptr, int row,
                                int col_tile, int stride) {
  /// reorder v to v_buffer
  /// v_buffer format [4][col_tile*4][64]
  int v_buffer_row = col_tile * 4;
  size_t v_stride = col_tile * 16 * 16;
  int v_real_step = (row + 3) / 4;
  int v_tail = row - (v_real_step - 1) * 4;
  auto v_ptr_ = reinterpret_cast<const int8_t(*)[stride]>(v_ptr);
  auto v_buffer_ = reinterpret_cast<int8_t(*)[v_buffer_row][64]>(v_buffer);

  for (int i = 0; i < v_buffer_row; i++) {
    if (i >= v_real_step - 1) {
      switch (v_tail) {
      case (1):
        tr_vnni_4x<1>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (2):
        tr_vnni_4x<2>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (3):
        tr_vnni_4x<3>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (4):
        tr_vnni_4x<4>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      default:
        tr_vnni_4x<0>(v_buffer_[0][i], v_ptr_[0], stride, v_stride);
        break;
      }
      v_tail = -1;
    } else {
      tr_vnni_4x<4>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
    }
  }
}

static void reorder_v_to_buffer_p64(int8_t *v_buffer, const int8_t *v_ptr, int row, int stride) {
  /// reorder v to v_buffer
  /// v_buffer format [4][col_tile*4][64]
  auto row_pad = (row + 63) / 64 * 64;
  int v_buffer_row = row_pad / 4;
  size_t v_stride = v_buffer_row * 64;
  int v_real_step = (row + 3) / 4;
  int v_tail = row - (v_real_step - 1) * 4;
  auto v_ptr_ = reinterpret_cast<const int8_t(*)[stride]>(v_ptr);
  auto v_buffer_ = reinterpret_cast<int8_t(*)[v_buffer_row][64]>(v_buffer);

  for (int i = 0; i < v_buffer_row; i++) {
    if (i >= v_real_step - 1) {
      switch (v_tail) {
      case (1):
        tr_vnni_4x<1>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (2):
        tr_vnni_4x<2>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (3):
        tr_vnni_4x<3>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      case (4):
        tr_vnni_4x<4>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
        break;
      default:
        tr_vnni_4x<0>(v_buffer_[0][i], v_ptr_[0], stride, v_stride);
        break;
      }
      v_tail = -1;
    } else {
      tr_vnni_4x<4>(v_buffer_[0][i], v_ptr_[i * 4], stride, v_stride);
    }
  }
}

// We limit row_tile 1 or 2, col_tile: 3, ..., 24 (384, could be more)
template <int row_tile, int col_tile> struct qk_gemm_impl {
  static constexpr int ldc = col_tile * 16;

  inline static void tile_loada(const void *a, size_t lda, int overlap) {
    auto a_ = reinterpret_cast<const int8_t(*)[lda]>(a);
    _tile_loadd(TMM4, a_[0], lda);
    if (row_tile == 2)
      _tile_loadd(TMM5, a_[16 - overlap], lda);
  }

  template <bool tail>
  inline static void tile_loadb(const void *b, int col_idx) {
    auto b_ = reinterpret_cast<const int8_t(*)[1024]>(b);
    _tile_loadd(TMM6, b_[col_idx * 2], 64);
    if (!tail)
      _tile_loadd(TMM7, b_[col_idx * 2 + 1], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
  }

  template <bool tail> inline static void dot_prod(void *c, void *com_c, int col_idx) {
    auto c_ = reinterpret_cast<int (*)[col_tile][16][16]>(c);
    auto com_c_ = reinterpret_cast<int(*)[ldc]>(com_c);

    __tile_dpbssd<TMM0, TMM4, TMM6>();
    _tile_stored(TMM0, c_[0][col_idx * 2], 64);
    _tile_stored(TMM0, &com_c_[0][col_idx * 32], ldc * 4);

    if (!tail) {
      __tile_dpbssd<TMM1, TMM4, TMM7>();
      _tile_stored(TMM1, c_[0][col_idx * 2 + 1], 64);
      _tile_stored(TMM1, &com_c_[0][col_idx * 32 + 16], ldc * 4);
    }

    if (row_tile == 2) {
      __tile_dpbssd<TMM2, TMM5, TMM6>();
      _tile_stored(TMM2, c_[1][col_idx * 2], 64);
      _tile_stored(TMM2, &com_c_[16][col_idx * 32], ldc * 4);

      if (!tail) {
        __tile_dpbssd<TMM3, TMM5, TMM7>();
        _tile_stored(TMM3, c_[1][col_idx * 2 + 1], 64);
        _tile_stored(TMM3, &com_c_[16][col_idx * 32 + 16], ldc * 4);
      }
    }
  }

  // Preloaded A
  inline static void compute(void *c, void *com_c, const void *a, const void *b, size_t lda,
                             int overlap) {
    constexpr int col_tail = col_tile % 2;
    constexpr int col_loop = col_tile / 2;

    tile_loada(a, lda, overlap);

    int i = 0;
#pragma unroll(col_loop)
    for (; i < col_loop; ++i) {
      tile_loadb<false>(b, i);
      zero_accum();
      dot_prod<false>(c, com_c, i);
    }

    if (col_tail) {
      tile_loadb<true>(b, i);
      zero_accum();
      dot_prod<true>(c, com_c, i);
    }
  }

  inline static void softmax(int8_t *c_int8, int *c, int *com_c, int len, float M,
                             float oscale) {
    assert(len <= col_tile * 16);
    auto c_ = reinterpret_cast<int (*)[col_tile][16][16]>(c);
    auto com_c_ = reinterpret_cast<int(*)[ldc]>(com_c);

    int8_t c_int8_com[32][col_tile * 16];
    // TODO: how many 16*64 tile does c_int8 have?
    // TODO: would col_tile must be divided by 4?
    int c_int8_p64_ctile = (col_tile * 16 + 63) / 64;
    auto c_len_pad64 = c_int8_p64_ctile * 64;
    auto c_int8_ = reinterpret_cast<int8_t (*)[c_int8_p64_ctile][16][64]>(c_int8);

    i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<16>::run(c_int8_[0], c_[0], len, M, oscale, c_len_pad64);
    i32_scale_attlen_softmax_scale_i8<16, 16>::run(c_int8_com, &com_c_[0][0], len, M, oscale, ldc);

    // for (int i = 0; i < col_tile; i++) {
    //   printf("compare old and new softmax: length %d: %d ~ %d!\n", len, i * 16, (i + 1) * 16);
    //   compare_matrix<int>((int*)c_[0][i], (int*)&com_c_[0][i * 16], 16, 16, 16, ldc);
    //   getchar();
    // }

    // for (int i = 0; i < c_int8_p64_ctile; i++) {
    //   printf("compare old and new softmax: length %d: %d ~ %d!\n", len, i * 64, (i + 1) * 64);
    //   compare_matrix<int8_t>((int8_t*)c_int8_[0][i], (int8_t*)&c_int8_com[0][i * 64], 16, 64, 64, ldc);
    //   getchar();
    // }
    
    // for (int i = 0; i < c_int8_p64_ctile; i++) {
    //   printf("---------------qk softmax output i: %d---------------\n", i);
    //   print_2d_matrix<int8_t>((int8_t*)c_int8_[0][i], 16, 64, 64);
    //   getchar();
    //   print_2d_matrix<int8_t>((int8_t*)&c_int8_com[0][i * 64], 16, 64, ldc);
    //   getchar();
    // }
    // print_2d_matrix<int>((int*)c_[0][0], 16, 16, 16);
    // getchar();
    // print_2d_matrix<int>((int*)c_[0][1], 16, 16, 16);
    // getchar();
    // print_2d_matrix<int8_t>((int8_t*)c_int8_[0][0], 16, 16, 64);
    // getchar();
    if (row_tile == 2) {
      i32_scale_attlen_softmax_scale_i8_amx_tile_vnni<16>::run(c_int8_[1], c_[1], len, M, oscale, c_len_pad64);
      i32_scale_attlen_softmax_scale_i8<16, 16>::run(&c_int8_com[16][0], &com_c_[16][0], len, M, oscale, ldc);
      // for (int i = 0; i < col_tile; i++) {
      //   printf("compare old and new softmax: %d ~ %d!\n", i * 16, (i + 1) * 16);
      //   compare_matrix<int>((int*)c_[1][i], (int*)&com_c_[16][i * 16], 16, 16, 16, ldc);
      //   getchar();
      // }
      // for (int i = 0; i < c_int8_p64_ctile; i++) {
      //   printf("compare old and new softmax: %d ~ %d!\n", i * 64, (i + 1) * 64);
      //   compare_matrix<int8_t>((int8_t*)c_int8_[1][i], (int8_t*)&c_int8_com[16][i * 64], 16, 64, 64, ldc);
      //   getchar();
      // }
    }
  }
};

// n_tile: 1 or 2
// k_tile: {1, 2, 3, 4, 5, 6, 8, 17, 19, 23}
template <int n_tile, int k_step> struct av_gemm_impl {
  inline static void loada(void *a, size_t idx) {
    auto a_ = reinterpret_cast<int8_t (*)[k_step][16][64]>(a);

    _tile_loadd(TMM4, a_[0][idx], 64);
    // printf("av loada---------------\n");
    // print_2d_matrix<int8_t>((int8_t*)a_[0][idx], 16, 16, 64);
    // getchar();
    if (n_tile == 2)
      _tile_loadd(TMM5, a_[1][idx], 64);
  }

  inline static void loadb(void *b_scratch, size_t ldb) {
    auto b_ = reinterpret_cast<int8_t(*)[ldb]>(b_scratch);
    _tile_loadd(TMM6, b_[0], 64);
    _tile_loadd(TMM7, b_[1], 64);
  }

  inline static void zero_accum() {
    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);
  }

  inline static void dot_prod() {
    __tile_dpbssd<TMM0, TMM4, TMM6>();
    __tile_dpbssd<TMM1, TMM4, TMM7>();
    if (n_tile == 2) {
      __tile_dpbssd<TMM2, TMM5, TMM6>();
      __tile_dpbssd<TMM3, TMM5, TMM7>();
    }
  }

  inline static void store_quant(void *c, int overlap, float m2) {
    alignas(64) int scratch[n_tile * 16 * 32];
    auto scratch_ = reinterpret_cast<int(*)[16][16]>(scratch);
    _tile_stored(TMM0, scratch_[0], 64);
    _tile_stored(TMM1, scratch_[1], 64);

    if (n_tile == 2) {
      _tile_stored(TMM2, scratch_[2], 64);
      _tile_stored(TMM3, scratch_[3], 64);
    }
    // print_2d_matrix<int>((int*)scratch_[0], 16, 16, 16);
    // getchar();

    // quant out to c
    auto vscale = _mm512_set1_ps(m2);
    auto c_out = reinterpret_cast<int8_t (*)[64]>(c);

#pragma unroll(n_tile)
    for (int i = 0; i < n_tile; i++) {
#pragma unroll(16)
      for (int j = 0; j < 16; j++) {
        auto pr = _mm512_loadu_si512(scratch_[2 * i][j]);
        auto prf = _mm512_cvtepi32_ps(pr);
        auto iout = _mm512_scale_minmax_i8_ps(vscale, prf);
        int out_row = (i == n_tile - 1) ? (i * 16 - overlap + j) : (i * 16 + j);
        _mm512_mask_cvtepi32_storeu_epi8(&c_out[out_row][0], 0xffff, iout);
        pr = _mm512_loadu_si512(scratch_[2 * i + 1][j]);
        prf = _mm512_cvtepi32_ps(pr);
        iout = _mm512_scale_minmax_i8_ps(vscale, prf);
        _mm512_mask_cvtepi32_storeu_epi8(&c_out[out_row][16], 0xffff, iout);
      }
    }
  }

  inline static void compute(void *c, void *a, void *b_scratch, size_t overlap, float m2) {

    zero_accum();

    auto b_ = reinterpret_cast<int8_t (*)[k_step][16][64]>(b_scratch);
    auto c_ = reinterpret_cast<int8_t (*)[64]>(c);

    size_t ldb = k_step * 16 * 64;

#pragma unroll(k_step)
    for (int i = 0; i < k_step; ++i) {
      loada(a, i);
      loadb(b_[0][i], ldb);
      dot_prod();
    }
    store_quant(&c_[0][0], overlap, m2);
    zero_accum();

#pragma unroll(k_step)
    for (int i = 0; i < k_step; ++i) {
      loada(a, i);
      loadb(b_[2][i], ldb);
      dot_prod();
    }
    store_quant(&c_[0][32], overlap, m2);
  }
};

void amx_per_head(const void *qkv_ptr, int ldqkv, void *a_ptr, size_t sl,
                      float M, float oscale, int32_t att_mask, float M2) {

  int sl_pad = (sl + 15) / 16 * 16;
  int col_tile = sl_pad / 16;

  alignas(64) int8_t k_scrach[16 * sl_pad * 4];
  alignas(64) int8_t v_scrach[sl_pad * 64];
  int sl_pad64 = (sl + 63) / 64 * 64;
  alignas(64) int8_t v_scrach_p64[sl_pad64 * 64];

  auto q = reinterpret_cast<const int8_t(*)[ldqkv]>(qkv_ptr);
  int qkv_dis = ldqkv / 3;
  reorder_k_to_buffer(k_scrach, &q[0][qkv_dis], sl, col_tile, ldqkv);
  reorder_v_to_buffer(v_scrach, &q[0][qkv_dis * 2], sl, col_tile, ldqkv);
  reorder_v_to_buffer_p64(v_scrach_p64, &q[0][qkv_dis * 2], sl, ldqkv);

  auto v_scrach_ = reinterpret_cast<int8_t (*)[sl_pad/4][64]>(v_scrach);
  auto v_scrach_p64_ = reinterpret_cast<int8_t (*)[sl_pad64/4][64]>(v_scrach_p64);

  int v_row = col_tile * 4;
  int rt_v = 16;
  for (; rt_v > 0; rt_v--) {
    if (v_row % rt_v == 0) {
      break;
    }
  }

  int cur_r_pos = 0;
  int row_loop = col_tile / 2;
  int rollback =
      (sl % max_tile_row != 0) ? max_tile_row - (sl % max_tile_row) : 0;
  bool is_even = (col_tile % 2 == 0);
  alignas(64) int a_scrach[32 * sl_pad];
  alignas(64) int a_com_scrach[32 * sl_pad];
  alignas(64) int8_t apro_scrach[32 * sl_pad64];
  auto a = reinterpret_cast<int8_t(*)[64]>(a_ptr);
  auto qk_tilecfg = Tilecfg();
  auto av_tilecfg = Tilecfg(rt_v);
  bool recfg_tile = false;
  qk_tilecfg.set_config();
  for (int i = 0; i < row_loop; i++) {
    int overlap = (is_even && i == row_loop - 1) ? rollback : 0;
    cur_r_pos = i * 32;
    switch (col_tile) {
    case (3):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 3>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 3>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 1>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (4):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 4>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 4>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 1>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (5):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 5>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 5>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (6):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 6>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 6>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (7):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 7>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 7>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (8):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 8>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 8>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (9):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 9>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                  overlap);
      qk_gemm_impl<2, 9>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (10):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 10>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 10>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (11):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 11>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 11>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (12):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 12>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 12>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (13):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 13>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 13>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (14):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 14>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 14>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (15):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 15>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 15>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (16):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 16>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 16>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (17):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 17>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 17>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, overlap, M2);
      break;
    case (18):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 18>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 18>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (19):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 19>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 19>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, overlap, M2);
      break;
    case (20):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 20>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 20>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 5>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (21):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 21>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 21>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (22):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 22>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 22>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    case (23):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 23>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 23>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, overlap, M2);
      break;
    case (24):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<2, 24>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv,
                                   overlap);
      qk_gemm_impl<2, 24>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<2, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, overlap, M2);
      break;
    }
  }
  cur_r_pos += 32 - rollback;
  if (!is_even) {
    switch (col_tile) {
    case (3):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 3>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 3>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 1>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (5):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 5>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 5>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (7):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 7>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 7>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 2>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (9):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 9>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 9>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 3>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (11):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 11>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 11>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (13):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 13>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 13>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (15):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 15>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 15>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (17):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 17>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 17>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 17>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, 0, M2);
      break;
    case (19):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 19>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 19>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 19>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, 0, M2);
      break;
    case (21):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 21>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 21>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 6>::compute(a[cur_r_pos], apro_scrach,
                                  v_scrach_p64, 0, M2);
      break;
    case (23):
      qk_tilecfg.set_config(recfg_tile);
      qk_gemm_impl<1, 23>::compute(a_scrach, a_com_scrach, q[cur_r_pos], k_scrach, ldqkv, 0);
      qk_gemm_impl<1, 23>::softmax(apro_scrach, a_scrach, a_com_scrach, att_mask, M, oscale);
      av_tilecfg.set_config(recfg_tile);
      av_gemm_impl<1, 23>::compute(a[cur_r_pos], apro_scrach,
                                   v_scrach_p64, 0, M2);
      break;
    default:
      break;
    }
  }
}

} // namespace intel_mlperf