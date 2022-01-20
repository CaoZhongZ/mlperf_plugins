#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>
#include <string.h>
#include <iostream>
#include <chrono>

#include "amx_mha.hpp"
#include "amx_tdpbssd.hpp"
#include "i_softmax_tpp.hpp"
#include "helper.hpp"
#include "el_common_intrin.hpp"

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)


#define TMM0	0
#define TMM1	1
#define TMM2	2
#define TMM3	3
#define TMM4	4
#define TMM5	5
#define TMM6	6
#define TMM7	7

using Time = std::chrono::high_resolution_clock;

namespace intel_mlperf {

static constexpr int max_sl = 384;
static constexpr int max_tile_row = 16;
static constexpr int max_tile_colsb = 64;
enum class status_t {success, failed};

static struct tilecfg {
	uint8_t palette;	    /* byte 0 */
	uint8_t start_row;	    /* byte 1 */
	char rsvd1[14];		    /* bytes 2-15 */
	uint16_t tile_colsb[8];	/* bytes 16-31 */
	char rsvd2[16];		    /* bytes 32-47 */
	uint8_t tile_rows[8];	/* bytes 48-55 */
	char rsvd3[8];		    /* bytes 56-63 */
} __attribute__((packed)) tilecfg;


inline bool amx_init() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    return true;
}

inline status_t configure_tile(struct tilecfg *cfg, int ntile, int row, int colsb)
{
    if (row <= 0 && colsb <= 0 && row > 16 && colsb > 64 && ntile > 7){
        return status_t::failed;
    }
    cfg->tile_rows[ntile] = row;
    cfg->tile_colsb[ntile] = colsb;
    return status_t::success;
}

status_t reorder_k_to_buffer_v2(const int8_t* k_ptr, const int8_t* v_ptr,
                                int8_t* k_buffer, int8_t* v_buffer,
                                int row, int row_pad, int stride)
{
    /// reorder k to k_buffer and v to v_buffer
    auto k_ptr_ = reinterpret_cast<const int (*)[stride/4]>(k_ptr);
    auto v_ptr_ = reinterpret_cast<const int8_t (*)[stride]>(v_ptr);
    auto k_buffer_ = reinterpret_cast<int (*)[row_pad]>(k_buffer);
    auto v_buffer_ = reinterpret_cast<int8_t (*)[256]>(v_buffer);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < row_pad; ++j) {
            k_buffer_[i][j] = j >= row ? 0 : k_ptr_[j][i];
        }
    }

    int v_buffer_row = row_pad / 4;
    for (int i = 0; i < v_buffer_row; i++) {
        for (int j = 0; j < 256; j++) {
            int ori_row = i * 4 + j % 4;
            int ori_col = j / 4;
            v_buffer_[i][j] = ori_row < row ? v_ptr_[ori_row][ori_col] : 0;
        }
    }

    return status_t::success;
}

// We limit row_tile 1 or 2, col_tile: 3, ..., 24 (384, could be more)
template <int row_tile, int col_tile>
struct qk_gemm_impl {
    static constexpr int ldb = col_tile * 16;
    static constexpr int lda = 3072;

    inline static void tile_loada(const void *a, int overlap) {
        auto a_ = reinterpret_cast<const int8_t (*)[lda]>(a);
        _tile_loadd(TMM4, &a_[0][0], lda);
        if (row_tile == 2) _tile_loadd(TMM5, &a_[16-overlap][0], lda);
    }

    template <bool tail>
    inline static void tile_loadb(const void *b, int col_idx) {
        auto b_ = reinterpret_cast<const int(*)[ldb]>(b);
        _tile_loadd(TMM6, &b_[0][col_idx*32], ldb*4);
        if (!tail) _tile_loadd(TMM7, &b_[0][col_idx*32+16], ldb*4);
    }

    inline static void zero_accum() {
        _tile_zero(TMM0);
        _tile_zero(TMM1);
        _tile_zero(TMM2);
        _tile_zero(TMM3);
    }

    template <bool tail> 
    inline static void dot_prod(void *c, int col_idx, int overlap) {
        auto c_ = reinterpret_cast<int (*)[ldb]>(c);
    
        __tile_dpbssd<TMM0, TMM4, TMM6>();
        _tile_stored(TMM0, &c_[0][col_idx*32], ldb*4);

        if (!tail) {
            __tile_dpbssd<TMM1, TMM4, TMM7>();
            _tile_stored(TMM1, &c_[0][col_idx*32+16], ldb*4);
        }

        if (row_tile == 2) {
            __tile_dpbssd<TMM2, TMM5, TMM6>();
            _tile_stored(TMM2, &c_[16-overlap][col_idx*32], ldb*4);

            if (!tail) {
                __tile_dpbssd<TMM3, TMM5, TMM7>();
                _tile_stored(TMM3, &c_[16-overlap][col_idx*32+16], ldb*4);
            }
        }
    }

    // Preloaded A
    inline static void compute (void* c, const void* a, const void* b, int overlap) {
        constexpr int col_tail = col_tile % 2;
        constexpr int col_loop = col_tile / 2;
        
        tile_loada(a, overlap);

        int i = 0;
#       pragma unroll (col_loop)
        for (; i < col_loop; ++i) {
            tile_loadb<false>(b, i);
            zero_accum();
            dot_prod<false>(c, i, overlap);
        }

        if (col_tail) {
            tile_loadb<true>(b, i);
            zero_accum();
            dot_prod<true>(c, i, overlap);
        }
    }

    inline static void softmax(
        void *c_int8, void *c, int len, float M, float oscale, int overlap) {
        assert(len <= col_tile * 16);
        auto c_int8_ = reinterpret_cast<int8_t (*)[ldb]>(c_int8);
        auto c_ = reinterpret_cast<int (*)[ldb]>(c);

        i32_scale_attlen_softmax_scale_i8<16, 16>::run(c_int8_[0], c_[0], len, M, oscale, ldb);

        if (row_tile == 2) {
            i32_scale_attlen_softmax_scale_i8<16, 16>::run(
                c_int8_[16 - overlap], c_[16 - overlap], len, M, oscale, ldb);
        }
    }

    inline static void config(struct tilecfg *cfg) {
        const int tile_num = 8;
        for (int i = 0; i < tile_num; i++) {
            cfg->tile_rows[i] = 16;
            cfg->tile_colsb[i] = 64;
        }
        cfg->palette = 1;
        _tile_release();
        _tile_loadconfig(cfg);
    }
};
// n_tile: 1 or 2
// k_tile: {1, 2, 3, 4, 5, 6, 8, 17, 19, 23}

template <int n_tile, int k_step>
struct av_gemm_impl {
    static constexpr int ldb = 64;
    static constexpr int lscratch = 256;
    static constexpr int ldc = 64;

    inline static void config(struct tilecfg *cfg, int r_v) {
        // tiles A:2 B:2 C:4 as qk
        // input A tiles
        configure_tile(cfg, 4, 16, r_v * 4);
        configure_tile(cfg, 5, 16, r_v * 4);
        // input B tiles
        configure_tile(cfg, 6, r_v, 64);
        configure_tile(cfg, 7, r_v, 64);
        // output tiles
        configure_tile(cfg, 0, 16, 64);
        configure_tile(cfg, 1, 16, 64);
        configure_tile(cfg, 2, 16, 64);
        configure_tile(cfg, 3, 16, 64);

        cfg->palette = 1;
        _tile_release();
        _tile_loadconfig(cfg);
    }

    inline static void loada(void* a, size_t lda, size_t overlap) {
        auto a_ = reinterpret_cast<int8_t (*)[lda]>(a);

        _tile_loadd(TMM4, a_[0], lda);
        if (n_tile == 2)
            _tile_loadd(TMM5, a_[16 - overlap], lda);
    }

    inline static void loadb(void *b_scratch) {
        auto b_ = reinterpret_cast<int (*)[16]>(b_scratch);
        _tile_loadd(TMM6, b_[0], lscratch);
        _tile_loadd(TMM7, b_[1], lscratch);
    }

    inline static void zero_accum() {
        _tile_zero(TMM0);
        _tile_zero(TMM1);
        _tile_zero(TMM2);
        _tile_zero(TMM3);
    }

    inline static void transpose_b(void* b_scratch, void *b, size_t real_k) {

    }

    inline static void dot_prod() {
        __tile_dpbssd<TMM0, TMM4, TMM6>();
        __tile_dpbssd<TMM1, TMM4, TMM7>();
        if (n_tile == 2) {
            __tile_dpbssd<TMM2, TMM5, TMM6>();
            __tile_dpbssd<TMM3, TMM5, TMM7>();
        }
    }

    inline static void store_quant(void* c, int overlap, float m2) {
        int scratch [n_tile * 16][32];
        _tile_stored(TMM0, &scratch[0][0], 32*4);
        _tile_stored(TMM1, &scratch[0][16], 32*4);
        if (n_tile == 2) {
            _tile_stored(TMM2, &scratch[16-overlap][0], 32*4);
            _tile_stored(TMM3, &scratch[16-overlap][16], 32*4);
        }

        // quant out to c
        auto vscale = _mm512_set1_ps(m2);
        auto c_out = reinterpret_cast<int8_t (*)[64]>(c);

        int q_rows = n_tile * 16 - overlap;
        for (int i = 0; i < q_rows; i++) {
            for (int j = 0; j < 32; j += 16) {
                auto pr = _mm512_loadu_si512(&scratch[i][j]);
                auto prf = _mm512_cvtepi32_ps(pr);
                auto iout = _mm512_scale_minmax_i8_ps(vscale, prf);
                _mm512_mask_cvtepi32_storeu_epi8(&c_out[i][j], 0xffff, iout);
            }
        }
    }

    inline static void compute(void* c, void *a, size_t lda, int tile_row, void* b_scratch, size_t overlap, float m2) {
        
        zero_accum();
        auto a_block = lda / k_step;
        auto b_block = a_block / 4;
        
        auto a_ = reinterpret_cast<int8_t (*)[a_block]>(a);
        auto b_ = reinterpret_cast<int8_t (*)[b_block][256]>(b_scratch);
        auto c_ = reinterpret_cast<int8_t (*)[64]>(c);

#       pragma unroll (k_step)
        for (int i = 0; i < k_step; ++i) {
            loada(a_[i], lda, overlap);
            loadb(b_[i]);
            dot_prod();
        }
        store_quant(&c_[0][0], overlap, m2);

#       pragma unroll (k_step)
        for (int i = 0; i < k_step; ++i) {
            loada(a_[i], lda, overlap);
            loadb(b_[i]);
            dot_prod();
        }
        store_quant(&c_[0][32], overlap, m2);
    }
};


status_t amx_per_head(const void* qkv_ptr, int ldqkv, void* a_ptr, 
                      int sl, float M, float oscale, int32_t att_mask, float M2) {

    int sl_pad = (sl + 15) / 16 * 16;
    int col_tile = sl_pad / 16;
    int v_row = sl_pad / 4;
    int qkv_dis = ldqkv / 3;
    
    int rt_v = 16;
    for (; rt_v > 0; rt_v--) { 
        if (v_row % rt_v == 0) {
            break;
        }
    }

    int8_t k_scrach[16*sl_pad*4];
    int8_t v_scrach[sl_pad*64];

    auto k_ = reinterpret_cast<int8_t* (*)[sl_pad*4]>(k_scrach);
    auto v_ = reinterpret_cast<int8_t* (*)[256]>(k_scrach);

    auto q = reinterpret_cast<const int8_t (*)[ldqkv]>(qkv_ptr);
    auto a = reinterpret_cast<int8_t* (*)[64]>(a_ptr);

    reorder_k_to_buffer_v2(&q[0][qkv_dis], &q[0][qkv_dis*2], k_scrach, v_scrach, sl, sl_pad, ldqkv);
    
    int rollback = (sl % max_tile_row != 0) ? max_tile_row - (sl % max_tile_row) : 0;
    bool is_even = (col_tile % 2) == 0;
    
    int a_scrach[32*sl_pad];
    int8_t apro_scrach[32*sl_pad];

    int cur_r_pos = 0;
    int row_loop = col_tile / 2;
    for (int i = 0; i < row_loop; i++) {
        int overlap = (is_even && i == row_loop - 1) ? rollback : 0;
        cur_r_pos = i * 32;
        switch (col_tile) {
        case (3):
            qk_gemm_impl<2, 3>::config(&tilecfg);
            qk_gemm_impl<2, 3>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 3>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 1>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 1>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (4):
            qk_gemm_impl<2, 4>::config(&tilecfg);
            qk_gemm_impl<2, 4>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 4>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 1>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 1>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (5):
            qk_gemm_impl<2, 5>::config(&tilecfg);
            qk_gemm_impl<2, 5>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 5>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 2>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 2>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (6):
            qk_gemm_impl<2, 6>::config(&tilecfg);
            qk_gemm_impl<2, 6>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 6>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 2>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 2>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (7):
            qk_gemm_impl<2, 7>::config(&tilecfg);
            qk_gemm_impl<2, 7>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 7>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 2>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 2>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (8):
            qk_gemm_impl<2, 8>::config(&tilecfg);
            qk_gemm_impl<2, 8>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 8>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 2>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 2>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (9):
            qk_gemm_impl<2, 9>::config(&tilecfg);
            qk_gemm_impl<2, 9>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 9>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 3>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 3>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (10):
            qk_gemm_impl<2, 10>::config(&tilecfg);
            qk_gemm_impl<2, 10>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 10>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 4>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (11):
            qk_gemm_impl<2, 11>::config(&tilecfg);
            qk_gemm_impl<2, 11>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 11>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 4>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (12):
            qk_gemm_impl<2, 12>::config(&tilecfg);
            qk_gemm_impl<2, 12>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 12>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 3>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 3>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (13):
            qk_gemm_impl<2, 13>::config(&tilecfg);
            qk_gemm_impl<2, 13>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 13>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 4>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (14):
            qk_gemm_impl<2, 14>::config(&tilecfg);
            qk_gemm_impl<2, 14>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 14>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 4>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (15):
            qk_gemm_impl<2, 15>::config(&tilecfg);
            qk_gemm_impl<2, 15>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 15>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 4>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (16):
            qk_gemm_impl<2, 16>::config(&tilecfg);
            qk_gemm_impl<2, 16>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 16>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 4>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (17):
            qk_gemm_impl<2, 17>::config(&tilecfg);
            qk_gemm_impl<2, 17>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 17>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 17>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 17>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (18):
            qk_gemm_impl<2, 18>::config(&tilecfg);
            qk_gemm_impl<2, 18>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 18>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 6>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 6>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (19):
            qk_gemm_impl<2, 19>::config(&tilecfg);
            qk_gemm_impl<2, 19>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 19>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 19>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 19>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (20):
            qk_gemm_impl<2, 20>::config(&tilecfg);
            qk_gemm_impl<2, 20>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 20>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 5>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 5>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (21):
            qk_gemm_impl<2, 21>::config(&tilecfg);
            qk_gemm_impl<2, 21>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 21>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 6>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 6>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (22):
            qk_gemm_impl<2, 22>::config(&tilecfg);
            qk_gemm_impl<2, 22>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 22>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 8>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 8>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (23):
            qk_gemm_impl<2, 23>::config(&tilecfg);
            qk_gemm_impl<2, 23>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            qk_gemm_impl<2, 23>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            av_gemm_impl<2, 23>::config(&tilecfg, rt_v);
            av_gemm_impl<2, 23>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        case (24):
            std::cout << "col_tile" << col_tile << std::endl;
            // qk_gemm_impl<2, 24>::config(&tilecfg);
            // qk_gemm_impl<2, 24>::compute(a_scrach, q[cur_r_pos], k_scrach, overlap);
            // qk_gemm_impl<2, 24>::softmax(apro_scrach, a_scrach, att_mask, M, oscale, overlap);
            // av_gemm_impl<2, 6>::config(&tilecfg, rt_v);
            // av_gemm_impl<2, 6>::compute(&a[cur_r_pos][0], apro_scrach, sl_pad, rt_v, v_scrach, overlap, M2);
            break;
        }
    }
    cur_r_pos += 32 - rollback;
    if (!is_even) {
        switch (col_tile) {
        case (3):
            qk_gemm_impl<1, 3>::config(&tilecfg);
            qk_gemm_impl<1, 3>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 3>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 1>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 1>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (5):
            qk_gemm_impl<1, 5>::config(&tilecfg);
            qk_gemm_impl<1, 5>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 5>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 2>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 2>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (7):
            qk_gemm_impl<1, 7>::config(&tilecfg);
            qk_gemm_impl<1, 7>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 7>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 2>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 2>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (9):
            qk_gemm_impl<1, 9>::config(&tilecfg);
            qk_gemm_impl<1, 9>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 9>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 3>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 3>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (11):
            qk_gemm_impl<1, 11>::config(&tilecfg);
            qk_gemm_impl<1, 11>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 11>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (13):
            qk_gemm_impl<1, 13>::config(&tilecfg);
            qk_gemm_impl<1, 13>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 13>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (15):
            qk_gemm_impl<1, 15>::config(&tilecfg);
            qk_gemm_impl<1, 15>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 15>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 4>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 4>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (17):
            qk_gemm_impl<1, 17>::config(&tilecfg);
            qk_gemm_impl<1, 17>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 17>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 17>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 17>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (19):
            qk_gemm_impl<1, 19>::config(&tilecfg);
            qk_gemm_impl<1, 19>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 19>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 19>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 19>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (21):
            qk_gemm_impl<1, 21>::config(&tilecfg);
            qk_gemm_impl<1, 21>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 21>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 6>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 6>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        case (23):
            qk_gemm_impl<1, 23>::config(&tilecfg);
            qk_gemm_impl<1, 23>::compute(a_scrach[0], q[cur_r_pos], k_scrach[0], 0);
            qk_gemm_impl<1, 23>::softmax(apro_scrach[0], a_scrach[0], att_mask, M, oscale, 0);
            av_gemm_impl<1, 23>::config(&tilecfg, rt_v);
            av_gemm_impl<1, 23>::compute(a[cur_r_pos], apro_scrach[0], sl_pad, rt_v, v_scrach[0], 0, M2);
            break;
        }
    }


}

at::Tensor amx_mha(
    const at::Tensor& qkv,
    const at::Tensor& att_mask,
    const at::Scalar& m1,
    const at::Scalar& oscale,
    const at::Scalar& m2) {
    
    // std::cout << "call amx_mha" << std::endl;
    auto qkv_sizes = qkv.sizes();
    assert(qkv_sizes.size() == 3);
    auto bs = qkv_sizes[0];
    auto sl = qkv_sizes[1];
    auto stride = qkv_sizes[2];

    auto qkv_block = stride / 3;
    int head_size = 64;
    int head_num = qkv_block / head_size;

    
    auto start = Time::now();
    auto amx_status = amx_init();
    auto init_during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
    
    if (!amx_status) {
        printf("amx init failed!\n");
        return qkv;
    }
    // config tile 
    char* _tc = (char*)(&tilecfg);
    for (int i = 0; i < 64; i++)
        _tc[i] = 0;

    // create attention tensor
    auto attention = at::empty({bs, head_num, sl, head_size}, at::TensorOptions().
                        dtype<int8_t>().memory_format(c10::MemoryFormat::Contiguous));

    auto att_ptr = reinterpret_cast<int8_t (*)[head_num][sl][head_size]>(attention.data_ptr());

    // Dynamic allocate k_buffer

    auto origin_ptr = reinterpret_cast<int8_t (*)[sl][3][head_num][head_size]>(qkv.data_ptr());
    auto att_mask_p = reinterpret_cast<int32_t *>(att_mask.data_ptr());

    int64_t copy_time = 0;
    int64_t amx_time = 0;
    auto loop_start = Time::now();
    auto other_during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
// # pragma omp parallel for collapse(2)
    for (int i = 0; i < bs; i++) // batch size
    {
        for (int j = 0; j < head_num; j++) // head num
        {
            // amx_status = amx_init();
            auto cur_q_ptr = &origin_ptr[i][0][0][j][0];
            auto cur_a_ptr = &att_ptr[i][j][0][0];
            
            auto amx_start = Time::now();
            amx_per_head(cur_q_ptr, stride, cur_a_ptr, sl, m1.toFloat(), oscale.toFloat(), att_mask_p[i], m2.toFloat());
            auto time_point_3 = Time::now();
            
            amx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - amx_start).count();
        }
    }
    auto loop_during = std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - loop_start).count();

    std::cout << "-----------init during : " << (float)init_during / 1000 / 1000 << " ms--------------" << std::endl;
    std::cout << "-----------amx time: " << (float)amx_time / 1000 / 1000 << " ms--------------" << std::endl;
    std::cout << "-----------other time: " << (float)other_during / 1000 / 1000 << " ms--------------" << std::endl;

    return attention;
}

}