#include <assert.h>
#include <cstdlib>

#include "amx_tdpbssd.hpp"
#include "el_common_intrin.hpp"
#include "i_mha_tpp.hpp"
#include "i_softmax_tpp.hpp"
#include "transpose.hpp"

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

template <typename T, int pad> T to_next(T x) {
  return (x + pad -1) / pad * pad;
}

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

static void tr_vnni_x16(int8_t *scratch, const int8_t *src, int row,
                                int stride) {
  // src format int8_t (*)[16][stride]
  auto src_ = reinterpret_cast<const int8_t(*)[16][stride]>(src);
  // scratch format : int8_t [col_tile][16][64]
  auto scratch_ = reinterpret_cast<int8_t(*)[16 * 64]>(scratch);

  auto n_tile = row / 16;
  auto tail = row % 16;

  for (int i = 0; i < n_tile; i++) {
    tr_vnni_x64<16>(scratch_[i], src_[i], stride, 64);
  }

  if (tail > 0) {
    decltype(tr_vnni_x64<1>)* tr_vnni_tbl [] = {
      tr_vnni_x64<1>, tr_vnni_x64<2>, tr_vnni_x64<3>, tr_vnni_x64<4>,
      tr_vnni_x64<5>, tr_vnni_x64<6>, tr_vnni_x64<7>, tr_vnni_x64<8>,
      tr_vnni_x64<9>, tr_vnni_x64<10>, tr_vnni_x64<11>, tr_vnni_x64<12>,
      tr_vnni_x64<13>, tr_vnni_x64<14>, tr_vnni_x64<15>, tr_vnni_x64<16>,
    };

    tr_vnni_tbl[tail](scratch[n_block], src[n_block], stride, 64);
  }
}

// dual transpose in essential
static void tr_vnni_4x(int8_t *scratch, const int8_t *src, int row,
    int stride) {
  auto v_ptr_ = reinterpret_cast<const int8_t(*)[4][stride]>(v_ptr);

  auto n_tile = row / 4;
  auto tail = row % 4;
  auto v_stride = 4 * (n_tile + (tail>0)) * 64;

  // scratch format int8_t [4][row_pad4][64]
  auto scratch_ = reinterpret_cast<int8_t(*)[64]>(scratch);

  for (int i = 0; i < n_tile; i++) {
    tr_vnni_4x<4>(scratch_[i], src_[i], stride, v_stride);
  }

  switch (tail) {
  case (1):
    tr_vnni_4x<1>(scratch_[n_tile], src_[n_tile], stride, v_stride);
    break;
  case (2):
    tr_vnni_4x<2>(scratch_[n_tile], src_[n_tile], stride, v_stride);
    break;
  case (3):
    tr_vnni_4x<3>(scratch_[n_tile], src_[n_tile], stride, v_stride);
    break;
  case (4):
    tr_vnni_4x<4>(scratch_[n_tile], src_[n_tile], stride, v_stride);
    break;
  default:
    break;
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

  template <bool tail> inline static void dot_prod(void *c, int col_idx) {
    auto c_ = reinterpret_cast<int(*)[ldc]>(c);

    __tile_dpbssd<TMM0, TMM4, TMM6>();
    _tile_stored(TMM0, &c_[0][col_idx * 32], ldc * 4);

    if (!tail) {
      __tile_dpbssd<TMM1, TMM4, TMM7>();
      _tile_stored(TMM1, &c_[0][col_idx * 32 + 16], ldc * 4);
    }

    if (row_tile == 2) {
      __tile_dpbssd<TMM2, TMM5, TMM6>();
      _tile_stored(TMM2, &c_[16][col_idx * 32], ldc * 4);

      if (!tail) {
        __tile_dpbssd<TMM3, TMM5, TMM7>();
        _tile_stored(TMM3, &c_[16][col_idx * 32 + 16], ldc * 4);
      }
    }
  }

  // Preloaded A
  inline static void compute(void *c, const void *a, const void *b, size_t lda,
                             int overlap) {
    constexpr int col_tail = col_tile % 2;
    constexpr int col_loop = col_tile / 2;

    tile_loada(a, lda, overlap);

    int i = 0;
#pragma unroll(col_loop)
    for (; i < col_loop; ++i) {
      tile_loadb<false>(b, i);
      zero_accum();
      dot_prod<false>(c, i);
    }

    if (col_tail) {
      tile_loadb<true>(b, i);
      zero_accum();
      dot_prod<true>(c, i);
    }
  }

  inline static void softmax(int8_t *c_int8, int *c, int len, float M,
                             float oscale) {
    assert(len <= col_tile * 16);
    auto l = (row_tile == 1) ? 16 : 32;

    f_i32_scale_softmax_scale_i8(c_int8, c, len, M, oscale, ldc, l);
  }
};

// n_tile: 1 or 2
//
template <int k_tile, int k_tile> struct av_gemm_impl {
  inline static void loada(void *a, size_t lda, size_t overlap) {
    auto a_ = reinterpret_cast<int8_t(*)[lda]>(a);

    _tile_loadd(TMM4, a_[0], lda);
    if (n_tile == 2)
      _tile_loadd(TMM5, a_[16], lda);
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

    // quant out to c
    auto vscale = _mm512_set1_ps(m2);
    auto c_out = reinterpret_cast<int8_t(*)[64]>(c);

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

  inline static void compute(void *c, void *a, void *b, size_t lda,
      size_t overlap, float m2) {

    zero_accum();
    int a_block = lda / k_step;
    int b_block = a_block / 4;

    auto a_ = reinterpret_cast<int8_t(*)[lda]>(a);
    auto b_ = reinterpret_cast<int8_t(*)[lda / 4][64]>(b_scratch);
    auto c_ = reinterpret_cast<int8_t(*)[64]>(c);

    size_t ldb = lda * 16;

#pragma unroll(k_step)
    for (int i = 0; i < k_step; ++i) {
      loada(&a_[0][i * a_block], lda, overlap);
      loadb(&b_[0][i * b_block][0], ldb);
      dot_prod();
    }
    store_quant(&c_[0][0], overlap, m2);
    zero_accum();

#pragma unroll(k_step)
    for (int i = 0; i < k_step; ++i) {
      loada(&a_[0][i * a_block], lda, overlap);
      loadb(&b_[2][i * b_block][0], ldb);
      dot_prod();
    }
    store_quant(&c_[0][32], overlap, m2);
  }
};

// XXX: 64 width 64 amx
struct i_amx_mha_tpp {
  i_amx_mha(size_t sequence_length, size_t att_len,
      float M, float oscale, float M2)
    : sequence_length_(sequence_length), att_len_(att_len), 
      sl_p16_(to_next<16>(sequence_legnth_)),
      sl_p64_(to_next<64>(sequence_length_)),
      overlap_(sl_p16 - sequence_length) {
    loop_block_ = sequence_length / 32;
    loop_tail_ =  sequence_length % 32;

    compute_block_ = compute_block_tbl_[0][sl_p16_/16];
    compute_tail_ = compute_block_tbl_[loop_tail_ < 16][sl_p16_/16];
  }

private:
  const size_t sequence_length_;
  const size_t att_len_;

  const size_t sl_p16_;
  const size_t sl_p64_;

  const size_t overlap_;

  size_t loop_block_;
  size_t loop_tail_

  const decltype(compute_block<2,3>) * compute_block_tbl [][22] = {
    {
      compute_block<2,3>, compute_block<2,4>, compute_block<2,5>,
      compute_block<2,6>, compute_block<2,7>, compute_block<2,8>,
      compute_block<2,9>, compute_block<2,10>, compute_block<2,11>,
      compute_block<2,12>, compute_block<2,13>, compute_block<2,14>,
      compute_block<2,15>, compute_block<2,16>, compute_block<2,17>,
      compute_block<2,18>, compute_block<2,19>, compute_block<2,20>,
      compute_block<2,21>, compute_block<2,22>, compute_block<2,23>,
      compute_block<2,24>
    }, {
      compute_block<1,3>, compute_block<1,4>, compute_block<1,5>, 
      compute_block<1,6>, compute_block<1,7>, compute_block<1,8>,
      compute_block<1,9>, compute_block<1,10>, compute_block<1,11>,
      compute_block<1,12>, compute_block<1,13>, compute_block<1,14>,
      compute_block<1,15>, compute_block<1,16>, compute_block<1,17>,
      compute_block<1,18>, compute_block<1,19>, compute_block<1,20>,
      compute_block<1,21>, compute_block<1,22>, compute_block<1,23>,
      compute_block<1,24>
    }
  };

  const decltype(compute_block<2,3>) * compute_block_;
  const decltype(compute_block<2,3>) * compute_tail_;

public:
  template <int row_tile, int col_tile>
  void compute_block(void* C, void* Q, void* K, void *V, size_t ld_att,
      float M, float oscale, float M2) {

    alignas(64) int32_t gemm_result[row_tile][sl_p16_];
    qk_gemm_impl<row_tile, col_tile>::compute(
        gemm_result, Q, K, ld_att, overlap_);

    alignas(64) int8_t softmax_result[row_tile][sl_p64_];
    qk_gemm_impl<row_tile, col_tile>::softmax(
        softmax_result, gemm_result, att_len, M, oscale);

    av_gemm_impl<row_tile, col_tile/4>::compute(
        C, softmax_result, V, sl_p64_, overlap_, M2);
  }

  void compute_head(void *C, const void *ATT, int ld_att, float M,
      float oscale, float M2) {

    enum {Q_ = 0, K_, V_};

    auto q = reinterpret_cast<const int8_t(*)[ld_att/3]>(ATT);
    alignas(64) int8_t k_scrach[sl_p16_ * 64];
    alignas(64) int8_t v_scrach[sl_p64_ * 64];

    // 48K
    tr_vnni_x16(k_scrach, q[K_], sl, ld_q);
    tr_vnni_4x(v_scrach, q[V_], sl, ld_q);

    // We use 2 * 2 block outer product scheme if possible
    int row_block = sl_ / 32;
    int row_tail = sl_ % 32;

    auto c = reinterpret_cast<int8_t(*)[32][64]>(C);
    Tilecfg().set_config();

    auto attention = reinterpret_cast<const int8_t(*)[32][ld_att]>(ATT);

    for (int i = 0; i < loop_block; i++) {
      compute_block_(c[i], attention[i], k_scratch, v_scratch, ld_att, M, oscale, M2);
    }

    if (loop_tail > 0) {
      compute_tail_(c[row_block], q[row_block], k_scratch, v_scratch, ld_att,
              M, oscale, M2);
    }
  }
};

} // namespace intel_mlperf
