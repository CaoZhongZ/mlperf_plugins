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

status_t mha_init_av_tile(struct tilecfg *cfg, int r_v) {

    // TODO: add function to config av
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

    return status_t::success;
}

status_t mha_init_qk_tile(struct tilecfg *cfg) {
    // all tile config to 16x64
    const int tile_num = 8;


    for (int i = 0; i < tile_num; i++) {
        cfg->tile_rows[i] = 16;
        cfg->tile_colsb[i] = 64;
    }

    cfg->palette = 1;
    _tile_release();
    _tile_loadconfig(cfg);

    return status_t::success;
}

status_t reorder_k_to_buffer(const int8_t* k_ptr, int8_t* k_buffer, int row, int row_pad, int col, int stride) {
    /*
    reorder k from sl*64 to 16*sl*4
    */

    if (row * col > 16 * row_pad * 4) {
        return status_t::failed;
    }

    // auto ptr = k_ptr;
    int ks = row_pad * 4;
    for (int i = 0; i < row; i++)
    {
        int kc = i * 4;
        for (int j = 0; j < col; j += 4)
        {
            int kr = j >> 2;
            
            k_buffer[kr*ks+kc] = k_ptr[i*stride+j];
            k_buffer[kr*ks+kc+1] = k_ptr[i*stride+j+1];
            k_buffer[kr*ks+kc+2] = k_ptr[i*stride+j+2];
            k_buffer[kr*ks+kc+3] = k_ptr[i*stride+j+3];
        }
    }
    // only for test
    return status_t::success;
}

status_t reorder_k_to_buffer_v2(const int8_t* k_ptr, const int8_t* v_ptr,
                                int8_t* k_buffer, int8_t* v_buffer,
                                int row, int row_pad, int col, int stride)
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

template<int Rows, int Steps>
struct av_gemm_post_impl {
    inline static void run (const void* apro, const void* v_buffer, void* r_buffer, void* c_ptr,
                           int32_t sl_pad, int32_t rollback, float m2);
};

template<int Steps>
struct av_gemm_post_impl<32, Steps> {
    inline static void run (const void* apro, const void* v_buffer, void* r_buffer, void* c_ptr,
                           int32_t sl_pad, int32_t rollback, float m2) {
        auto a = reinterpret_cast<const int8_t (*)[sl_pad]>(apro);
        auto v = reinterpret_cast<const int8_t (*)[256]>(v_buffer);
        auto r = reinterpret_cast<const int (*)[64]>(r_buffer);

        int ac_block = sl_pad / Steps;
        int vr_block = ac_block / 4;

        _tile_zero(TMM0);
        _tile_zero(TMM1);
        _tile_zero(TMM2);
        _tile_zero(TMM3);

#       pragma unroll Steps
        for (int i = 0; i < Steps; i++) {
            int a_c = i * ac_block;
            int v_r = i * vr_block;
            _tile_loadd(TMM4, &a[0][a_c], sl_pad);
            _tile_loadd(TMM5, &a[16-rollback][a_c], sl_pad);
            _tile_loadd(TMM6, &v[v_r][0], 256);
            _tile_loadd(TMM7, &v[v_r][64], 256);

            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM4, TMM7>();
            __tile_dpbssd<TMM2, TMM5, TMM6>();
            __tile_dpbssd<TMM3, TMM5, TMM7>();
        }

        _tile_stored(TMM0, &r[0][0], 256);
        _tile_stored(TMM1, &r[0][16], 256);
        _tile_stored(TMM2, &r[16-rollback][0], 256);
        _tile_stored(TMM3, &r[16-rollback][16], 256);

        _tile_zero(TMM0);
        _tile_zero(TMM1);
        _tile_zero(TMM2);
        _tile_zero(TMM3);

#       pragma unroll Steps
        for (int i = 0; i < Steps; i++) {
            int a_c = i * ac_block;
            int v_r = i * vr_block;
            _tile_loadd(TMM4, &a[0][a_c], sl_pad);
            _tile_loadd(TMM5, &a[16-rollback][a_c], sl_pad);
            _tile_loadd(TMM6, &v[v_r][128], 256);
            _tile_loadd(TMM7, &v[v_r][192], 256);

            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM4, TMM7>();
            __tile_dpbssd<TMM2, TMM5, TMM6>();
            __tile_dpbssd<TMM3, TMM5, TMM7>();
        }

        _tile_stored(TMM0, &r[0][32], 256);
        _tile_stored(TMM1, &r[0][48], 256);
        _tile_stored(TMM2, &r[16-rollback][32], 256);
        _tile_stored(TMM3, &r[16-rollback][48], 256);

        // mul m2
        auto vscale = _mm512_set1_ps(m2);
        auto c_out = reinterpret_cast<int8_t (*)[64]>(c_ptr);

        for (int i = 0; i < 32-rollback; i ++) {
            for (int j = 0; j < 64; j += 16) {
                auto pr = _mm512_loadu_si512(&r[i][j]);
                auto prf = _mm512_cvtepi32_ps(pr);
                auto iout = _mm512_scale_minmax_i8_ps(vscale, prf);
                _mm512_mask_cvtepi32_storeu_epi8(&c_out[i][j], 0xffff, iout);
            }
        }
    }
};

template<int Steps>
struct av_gemm_post_impl<16, Steps> {
    inline static void run (const void* apro, const void* v_buffer, void* r_buffer, void* c_ptr,
                           int32_t sl_pad, int32_t rollback, float m2) {
        auto a = reinterpret_cast<const int8_t (*)[sl_pad]>(apro);
        auto v = reinterpret_cast<const int8_t (*)[256]>(v_buffer);
        auto r = reinterpret_cast<const int (*)[64]>(r_buffer);

        int ac_block = sl_pad / Steps;
        int vr_block = ac_block / 4;

        _tile_zero(TMM0);
        _tile_zero(TMM1);

#       pragma unroll Steps
        for (int i = 0; i < Steps; i++) {
            int a_c = i * ac_block;
            int v_r = i * vr_block;
            _tile_loadd(TMM4, &a[0][a_c], sl_pad);
            _tile_loadd(TMM6, &v[v_r][0], 256);
            _tile_loadd(TMM7, &v[v_r][64], 256);

            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM4, TMM7>();
        }
        _tile_stored(TMM0, &r[0][0], 256);
        _tile_stored(TMM1, &r[0][16], 256);

        _tile_zero(TMM0);
        _tile_zero(TMM1);

#       pragma unroll Steps
        for (int i = 0; i < Steps; i++) {
            int a_c = i * ac_block;
            int v_r = i * vr_block;
            _tile_loadd(TMM4, &a[0][a_c], sl_pad);
            _tile_loadd(TMM6, &v[v_r][128], 256);
            _tile_loadd(TMM7, &v[v_r][192], 256);

            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM4, TMM7>();
        }
        _tile_stored(TMM0, &r[0][32], 256);
        _tile_stored(TMM1, &r[0][48], 256);

        // mul m2
        auto vscale = _mm512_set1_ps(m2);

        auto c_out = reinterpret_cast<int8_t (*)[64]>(c_ptr);

        for (int i = 0; i < 16; i ++) {
            for (int j = 0; j < 64; j += 16) {
                auto pr = _mm512_loadu_si512(&r[i][j]);
                auto prf = _mm512_cvtepi32_ps(pr);
                auto iout = _mm512_scale_minmax_i8_ps(vscale, prf);
                _mm512_mask_cvtepi32_storeu_epi8(&c_out[i][j], 0xffff, iout);
            }
        }
    }
};

template<int rows>
inline void av_gemm_kernel(const void* apro, const void* v_buffer, void* r_buffer, void* c_ptr,
                           int32_t sl_pad, int32_t rollback, int32_t step, float m2);

template<>
inline void av_gemm_kernel<32>(const void* apro, const void* v_buffer, void* r_buffer, void* c_ptr,
                           int32_t sl_pad, int32_t rollback, int32_t step, float m2) {
    // do 32 row gemm
    // a shape [32, sl_pad]
    // v shape [sl_pad/4, 256]
    // r shape [32-rollback, 64]
    // steps : {1, 2, 3, 4, 5, 6, 8, 17, 19, 23}

    switch (step) {
    case (1):
        av_gemm_post_impl<32, 1>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (2):
        av_gemm_post_impl<32, 2>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (3):
        av_gemm_post_impl<32, 3>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (4):
        av_gemm_post_impl<32, 4>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (5):
        av_gemm_post_impl<32, 5>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (6):
        av_gemm_post_impl<32, 6>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (8):
        av_gemm_post_impl<32, 8>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (17):
        av_gemm_post_impl<32, 17>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (19):
        av_gemm_post_impl<32, 19>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (23):
        av_gemm_post_impl<32, 23>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    }
}

template<>
inline void av_gemm_kernel<16>(const void* apro, const void* v_buffer, void* r_buffer, void* c_ptr,
                           int32_t sl_pad, int32_t rollback, int32_t step, float m2) {
    // do 32 row gemm
    // a shape [32, sl_pad]
    // v shape [sl_pad/4, 256]
    // r shape [32-rollback, 64]
    // steps : {1, 2, 3, 4, 5, 6, 8, 17, 19, 23}
    // TODO: unroll steps

    switch (step) {
    case (1):
        av_gemm_post_impl<16, 1>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (2):
        av_gemm_post_impl<16, 2>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (3):
        av_gemm_post_impl<16, 3>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (4):
        av_gemm_post_impl<16, 4>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (5):
        av_gemm_post_impl<16, 5>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (6):
        av_gemm_post_impl<16, 6>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (8):
        av_gemm_post_impl<16, 8>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (17):
        av_gemm_post_impl<16, 17>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (19):
        av_gemm_post_impl<16, 19>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    case (23):
        av_gemm_post_impl<16, 23>::run(apro, v_buffer, r_buffer, c_ptr, sl_pad, rollback, m2);
        break;
    }
}


// We limit row_tile 1 or 2, col_tile: 3, ..., 24 (384, could be more)
template <int row_tile, int col_tile>
struct qk_gemm_impl {
    constexpr auto ldb = col_tile * 16;
    constexpr auto lda = 3072;

    inline static tile_loada(const void *a, int overlap) {
        auto a_ = reinterpret_cast<const int(*)[lda]>(a);
        _tile_loadd(TMM4, a_[0], lda);
        if (row_tile == 2) _tile_loadd(TMM5, a_[16-overlap], lda);
    }

    template <bool tail>
    inline static tile_loadb(const void *b, int col_idx) {
        auto b_ = reinterpret_cast<const int(*)[16]>(b);
        _tile_loadd(TMM6, b_[col_idx], ldb*4);
        if (!tail) _tile_loadd(TMM7, b_[col_idx + 1], ldb*4);
    }

    inline static zero_accum() {
        _tile_zero(TMM0);
        _tile_zero(TMM1);
        _tile_zero(TMM2);
        _tile_zero(TMM3);
    }

    template <bool tail> inline static dot_prod(void *c, int col_idx, int overlap) {
        auto c_ = reinterpret_cast<int (*)[ldb/16][16]>(c);
    
        __tile_dpbssd<TMM0, TMM4, TMM6>();
        _tile_stored(TMM0, c_[0][col_idx], ldb*4);

        if (!tail) {
            __tile_dpbssd<TMM1, TMM4, TMM7>();
            _tile_stored(TMM1, c_[0][col_idx+1], ldb*4);
        }

        if (row_tile == 2) {
            __tile_dpbssd<TMM2, TMM5, TMM6>();
            _tile_stored(TMM2, c_[16-overlap][col_idx], ldb*4);

            if (!tail) {
                __tile_dpbssd<TMM3, TMM5, TMM7>();
                _tile_stored(TMM3, c_[16-overlap][col_idx+1], ldb*4);
            }
        }
    }

    // Preloaded A
    inline static void compute (void* c, const void* b, const void* a, int overlap) {
        constexpr auto col_tail = col_tile % 2;
        constexpr auto col_loop = (col_tile - col_tail)/2;
        tile_loada(a, overlap);

        int i = 0;
#       pragma unroll (col_loop)
        for (; i < col_loop; ++i) {
            tile_loadb<false>(b, i*2);
            zero_accum();
            dot_prod<false>(c, i*2, overlap);
        }

        if (col_tail) {
            tile_loadb<true>(b, i*2);
            zero_accum();
            dot_prod<true>(c, i*2, overlap);
        }
    }

    inline static void softmax(
        void *c_int8, void *c, int len, float M, float oscale, int overlap) {
        auto actual_row = row_tile * 16 - overlap;
        assert(len <= col_tile * 16);
        auto c_int8_ = reinterpret_cast<int8_t (*)[ldb]>(c_int8);
        auto c_ = reinterpret_cast<int (*)[ldb]>(c);

        i32_scale_attlen_softmax_scale_i8<16, 16>(c_int8_[0], c_[0], len, M, oscale, ldb);

        if (row_tile == 2) {
            i32_scale_attlen_softmax_scale_i8<16, 16>(
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
    constexpr size_t ldb = 64;
    constexpr size_t lscratch = 256;
    constexpr size_t ldc = 64;

    inline void config(struct tilecfg *cfg, int r_v) {
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

    inline void loada(void* a, size_t lda, size_t overlap) {
        auto a_ = reinterpret_cast<int8_t (*)[lda]>(a);

        _tile_loadd(TMM4, a_[0], lda);
        if (n_tile == 2)
            _tile_loadd(TMM5, a[16 - overlap], lda);
    }

    inline void loadb(void *b_scratch) {
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

    inline transpose_b(void* b_scratch, void *b, size_t real_k) {

    }

    inline void dot_prod() {
        __tile_dpbssd<TMM0, TMM4, TMM6>();
        __tile_dpbssd<TMM1, TMM4, TMM7>();
        if (n_tile == 2) {
            __tile_dpbssd<TMM2, TMM5, TMM6>();
            __tile_dpbssd<TMM3, TMM5, TMM7>();
        }
    }

    inline void store_quant(void *c, size_t overlap, float m2) {
        alignas(64) int scratch [n_tile][32];
        _tile_stored(TMM0, scratch[0], 32*4);
        _tile_stored(TMM1, scratch[1], 32*4);
        if (n_tile == 2) {
            _tile_stored(TMM2, scratch[16-overlap][0], 32*4);
            _tile_stored(TMM3, scratch[16-overlap][1], 32*4);
        }

        // quant out to c
        auto vscale = _mm512_set1_ps(m2);
        auto c_out = reinterpret_cast<int8_t (*)[64]>(c);

        auto q_rows = n_tile * 16 - overlap;
        for (int i = 0; i < q_rows; i ++) {
            for (int j = 0; j < 32; j += 16) {
                auto pr = _mm512_loadu_si512(&scratch[i][j]);
                auto prf = _mm512_cvtepi32_ps(pr);
                auto iout = _mm512_scale_minmax_i8_ps(vscale, prf);
                _mm512_mask_cvtepi32_storeu_epi8(&c_out[i][j], 0xffff, iout);
            }
        }
    }

    inline void compute(void* c, void *a, size_t lda, void* b_scratch, size_t overlap, float m2) {
        zero_accum();

        auto b_block = lda / k_step;
        auto a_block = b_block * 4;
        
        auto a_ = reinterpret_cast<int8_t (*)[a_block>();
        auto b_ = reinterpret_cast<int8_t (*)[b_block][64]>(b_scratch);
        auto c_ = reinterpret_cast<int8_t (*)[16]>(c);

#       pragma unroll (k_step)
        for (int i = 0; i < k_step; ++i) {
            this->loada(a_[i], lda, overlap);
            this->loadb(b_[i]);
            dot_prod();
        }
        store_quant(c_[0], m2);

#       pragma unroll (k_step)
        for (int i = 0; i < k_step; ++i) {
            this->loada(a_[i], lda, overlap);
            this->loadb(b_[i]);
            dot_prod();
        }
        store_quant(c_[2], m2);
    }
};


status_t amx_per_head(const void* qkv_ptr, int ldqkv, void* a_ptr, int lda,
                      int sl, float M, float oscale, int32_t att_mask, float M2) {

    int sl_pad = (sl + 15) / 16 * 16;
    int col_tile = sl_pad / 16;
    int v_row = sl_pad / 4;
    int rt_v = 16;
    for (; rt_v > 0; rt_v--) { 
        if (v_row % rt_v == 0) {
            break;
        }
    }
    int k_step = v_row / rt_v; 
    int8_t k_scrach[16][sl_pad*4];
    int8_t v_scrach[sl_pad/4][256];

    auto q = reinterpret_cast<const int8_t (*)[lda]>(qkv_ptr);
    auto a = reinterpret_cast<int8_t* (*)[64]>(a_ptr);


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
    mha_init_qk_tile(&tilecfg);

    // create attention tensor
    int nbq_row = (sl + 15) / max_tile_row;
    int sl_pad = max_tile_row * nbq_row;
    auto attention = at::empty({bs, head_num, sl, head_size}, at::TensorOptions().
                        dtype<int8_t>().memory_format(c10::MemoryFormat::Contiguous));

    auto att_ptr = reinterpret_cast<int8_t (*)[head_num][sl][head_size]>(attention.data_ptr());

    // Dynamic allocate k_buffer
    int8_t k_buffer[sl_pad*64];
    int att_buffer[sl_pad*2*max_tile_row];
    uint8_t att_pro_buffer[sl_pad*2*max_tile_row];
    int8_t v_buffer[sl_pad*64];
    int r_buffer[2*max_tile_row*64];

    auto origin_ptr = reinterpret_cast<int8_t (*)[sl][3][head_num][head_size]>(qkv.data_ptr());
    auto att_mask_p = reinterpret_cast<int32_t *>(att_mask.data_ptr());
    int rollback = (sl % max_tile_row != 0) ? max_tile_row - (sl % max_tile_row) : 0;

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
            auto cur_k_ptr = &origin_ptr[i][0][1][j][0];
            auto cur_v_ptr = &origin_ptr[i][0][2][j][0];
            auto cur_a_ptr = &att_ptr[i][j][0][0];
            
            auto time_point_1 = Time::now();
            reorder_k_to_buffer_v2(cur_k_ptr, cur_v_ptr, 
                                   k_buffer, v_buffer, 
                                   sl, sl_pad, head_size, stride);
            auto time_point_2 = Time::now();
            copy_time += std::chrono::duration_cast<std::chrono::nanoseconds>(time_point_2 - time_point_1).count();

            amx_per_head(cur_q_ptr, k_buffer, att_buffer, v_buffer, att_pro_buffer, r_buffer, cur_a_ptr, nbq_row, stride,
                                max_tile_row, max_tile_row, rollback, 
                                m1.toFloat(), oscale.toFloat(), att_mask_p[i], m2.toFloat());
            auto time_point_3 = Time::now();
            
            amx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(time_point_3 - time_point_2).count();
        }
    }
    auto loop_during = std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - loop_start).count();

    std::cout << "init during : " << (float)init_during / 1000 / 1000 << " ms" << std::endl;
    std::cout << "-----------copy time: " << (float)copy_time / 1000 / 1000 << " ms--------------" << std::endl;
    std::cout << "-----------amx time: " << (float)amx_time / 1000 / 1000 << " ms--------------" << std::endl;
    std::cout << "-----------loop time: " << loop_during << " ms--------------" << std::endl;
    std::cout << "-----------other time: " << (float)other_during / 1000 / 1000 << " ms--------------" << std::endl;

    return attention;
}

}