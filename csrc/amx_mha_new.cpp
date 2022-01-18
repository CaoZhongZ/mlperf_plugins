#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>
#include <string.h>
#include <iostream>

#include "amx_mha.hpp"
#include "amx_tdpbssd.hpp"
#include "i_softmax_tpp.hpp"
#include "helper.hpp"
#include "el_common_intrin.hpp"

namespace intel_mlperf {

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
    const int tile_config_size_in_bytes = 64;
    const int tile_num = 8;

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
    const int tile_config_size_in_bytes = 64;
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
    int nc_block = col / 4;
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
            v_buffer_[i][j] = i * 4 * 64 + j < row * 64 ? v_ptr_[ori_row][ori_col] : 0;
        }
    }

    return status_t::success;
}

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
    // TODO: unroll steps
    auto a = reinterpret_cast<const int8_t (*)[sl_pad]>(apro);
    auto v = reinterpret_cast<const int8_t (*)[256]>(v_buffer);
    auto r = reinterpret_cast<const int (*)[64]>(r_buffer);

    int ac_block = sl_pad / step;
    int vr_block = ac_block / 4;

    _tile_zero(TMM0);
    _tile_zero(TMM1);
    _tile_zero(TMM2);
    _tile_zero(TMM3);

    for (int i = 0; i < step; i++) {
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

    for (int i = 0; i < step; i++) {
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

template<>
inline void av_gemm_kernel<16>(const void* apro, const void* v_buffer, void* r_buffer, void* c_ptr,
                           int32_t sl_pad, int32_t rollback, int32_t step, float m2) {
    // do 16 row gemm
    // a shape [32, sl_pad]
    // v shape [sl_pad/4, 256]
    // r shape [sl, 64]
    // steps : {1, 2, 3, 4, 5, 6, 8, 17, 19, 23}
    // TODO: unroll steps
    auto a = reinterpret_cast<const int8_t (*)[sl_pad]>(apro);
    auto v = reinterpret_cast<const int8_t (*)[256]>(v_buffer);
    auto r = reinterpret_cast<const int (*)[64]>(r_buffer);

    int ac_block = sl_pad / step;
    int vr_block = ac_block / 4;

    _tile_zero(TMM0);
    _tile_zero(TMM1);
    for (int i = 0; i < step; i++) {
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
    for (int i = 0; i < step; i++) {
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


template <int even, int NBlock>
struct amx_qk_gemm_softmax_impl {
    inline static void run (const void* q_ptr, const void* k_ptr,
                            void* a_buffer, void* v_buffer, void* apro_buffer, void* r_buffer, void* a_ptr, 
                            int ldq, int q_block, int k_block, int rollback, 
                            float M, float oscale, int32_t att_mask, float M2);
};

template <int NBlock>
struct amx_qk_gemm_softmax_impl<1, NBlock> {
    inline static void run (const void* q_ptr, const void* k_ptr,
                            void* a_buffer, void* v_buffer, void* apro_buffer, void* r_buffer, void* a_ptr, 
                            int ldq, int q_block, int k_block, int rollback, 
                            float M, float oscale, int32_t att_mask, float M2) {
        // sl / 16 is even

        int sl_pad = NBlock * 32;
        int vbuffer_r = NBlock * 8;
        int rt_v = max_tile_row;
        for (; rt_v > 0; rt_v--) { 
            if (vbuffer_r % rt_v == 0) {
                break;
            }
        }
        int step = vbuffer_r / rt_v;
        auto q = reinterpret_cast<const int (*)[ldq/4]>(q_ptr);
        auto k = reinterpret_cast<const int (*)[sl_pad]>(k_ptr);
        auto a = reinterpret_cast<int (*)[sl_pad]>(a_buffer);
        auto r = reinterpret_cast<int (*)[64]>(r_buffer);
        auto c = reinterpret_cast<int8_t (*)[64]>(a_ptr);

        int q_s = ldq;
        int k_s = sl_pad * 4;
        int a_s = k_s;

        i_softmax_tpp<16> softmax_full(1, 1, 32, sl_pad);
        i_softmax_tpp<16> softmax_tail(1, 1, 32-rollback, sl_pad);


#       pragma unroll NBlock
        for (int i = 0; i < NBlock; i++) {
            int cur_q_pos = i * max_tile_row * 2;
            int rollback_ = (i == NBlock - 1) ? rollback : 0;
            int cur_q_pos_b = cur_q_pos + q_block - rollback_;

            _tile_loadd(TMM4, &q[cur_q_pos][0], q_s);
            _tile_loadd(TMM5, &q[cur_q_pos_b][0], q_s);
#           pragma unroll NBlock
            for (int j = 0; j < NBlock; j++) {
                int cur_k_pos = j * k_block * 2;
                int cur_k_pos_r = cur_k_pos + k_block;

                _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);
                _tile_loadd(TMM7, &k[0][cur_k_pos_r], k_s);

                _tile_zero(TMM0);
                _tile_zero(TMM1);
                _tile_zero(TMM2);
                _tile_zero(TMM3);
                
                __tile_dpbssd<TMM0, TMM4, TMM6>();
                __tile_dpbssd<TMM1, TMM4, TMM7>();
                __tile_dpbssd<TMM2, TMM5, TMM6>();
                __tile_dpbssd<TMM3, TMM5, TMM7>();

                _tile_stored(TMM0, &a[0][cur_k_pos], a_s);
                _tile_stored(TMM1, &a[0][cur_k_pos_r], a_s);
                _tile_stored(TMM2, &a[q_block-rollback_][cur_k_pos], a_s);
                _tile_stored(TMM3, &a[q_block-rollback_][cur_k_pos_r], a_s);
            }
            // add softmax
            auto& softmax_computer = (i == NBlock - 1) ? softmax_tail : softmax_full;
            softmax_computer.ref(apro_buffer, a_buffer, &att_mask, M, oscale);
            
            // add av gemm
            mha_init_av_tile(&tilecfg, rt_v);
            av_gemm_kernel<32>(apro_buffer, v_buffer, &r[cur_q_pos][0], &c[cur_q_pos][0], sl_pad, rollback_, step, M2);
            mha_init_qk_tile(&tilecfg);
        }
    }
};

template<int NBlock>
struct amx_qk_gemm_softmax_impl<0, NBlock> {
    inline static void run (const void* q_ptr, const void* k_ptr,
                            void* a_buffer, void* v_buffer, void* apro_buffer, void* r_buffer, void* a_ptr, 
                            int ldq, int q_block, int k_block, int rollback, 
                            float M, float oscale, int32_t att_mask, float M2) {
        // sl / 16 is odd
        
        int sl_pad = NBlock * 32 + 16;
        int vbuffer_r = NBlock * 8 + 4;
        int rt_v = max_tile_row;
        for (; rt_v > 0; rt_v--) { 
            if (vbuffer_r % rt_v == 0) {
                break;
            }
        }
        int step = vbuffer_r / rt_v;
        auto q = reinterpret_cast<const int (*)[ldq/4]>(q_ptr);
        auto k = reinterpret_cast<const int (*)[sl_pad]>(k_ptr);
        auto a = reinterpret_cast<int (*)[sl_pad]>(a_buffer);
        auto r = reinterpret_cast<int (*)[64]>(r_buffer);
        auto c = reinterpret_cast<int8_t (*)[64]>(a_ptr);

        int q_s = ldq;
        int k_s = sl_pad * 4;
        int a_s = k_s;

        int cur_q_pos = 0;
        int cur_q_pos_b = 0;
        int cur_k_pos = 0;
        int cur_k_pos_r = 0;

        i_softmax_tpp<16> softmax_full(1, 1, 32, sl_pad); 
        i_softmax_tpp<16> softmax_tail(1, 1, 16, sl_pad);

#       pragma unroll NBlock
        for (int i = 0; i < NBlock; i++) {
            cur_q_pos = i * q_block * 2;
            cur_q_pos_b = cur_q_pos + q_block;

            _tile_loadd(TMM4, &q[cur_q_pos][0], q_s);
            _tile_loadd(TMM5, &q[cur_q_pos_b][0], q_s);

#           pragma unroll NBlock
            for (int j = 0; j < NBlock; j++) {
                cur_k_pos = j * k_block * 2;
                cur_k_pos_r = cur_k_pos + k_block;
                // coordinate of A

                _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);
                _tile_loadd(TMM7, &k[0][cur_k_pos_r], k_s);

                _tile_zero(TMM0);
                _tile_zero(TMM1);
                _tile_zero(TMM2);
                _tile_zero(TMM3);
                
                __tile_dpbssd<TMM0, TMM4, TMM6>();
                __tile_dpbssd<TMM1, TMM4, TMM7>();
                __tile_dpbssd<TMM2, TMM5, TMM6>();
                __tile_dpbssd<TMM3, TMM5, TMM7>();

                _tile_stored(TMM0, &a[0][cur_k_pos], a_s);
                _tile_stored(TMM1, &a[0][cur_k_pos_r], a_s);
                _tile_stored(TMM2, &a[q_block][cur_k_pos], a_s);
                _tile_stored(TMM3, &a[q_block][cur_k_pos_r], a_s);
            }

            cur_k_pos += k_block * 2;

            _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);

            _tile_zero(TMM0);
            _tile_zero(TMM1);
            
            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM5, TMM6>();

            _tile_stored(TMM0, &a[0][cur_k_pos], a_s);
            _tile_stored(TMM1, &a[q_block][cur_k_pos], a_s);

            // add softmax
            softmax_full.ref(apro_buffer, a_buffer, &att_mask, M, oscale);

            // av gemm
            mha_init_av_tile(&tilecfg, rt_v);
            av_gemm_kernel<32>(apro_buffer, v_buffer, &r[cur_q_pos][0], &c[cur_q_pos][0], sl_pad, 0, step, M2);
            mha_init_qk_tile(&tilecfg);
        }

        // the last 16 row
        cur_q_pos += q_block * 2 - rollback;
        _tile_loadd(TMM4, &q[cur_q_pos][0], q_s);

#       pragma unroll NBlock
        for (int j = 0; j < NBlock; j++) {
            cur_k_pos = j * k_block * 2;
            cur_k_pos_r = cur_k_pos + k_block;

            _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);
            _tile_loadd(TMM7, &k[0][cur_k_pos_r], k_s);

            _tile_zero(TMM0);
            _tile_zero(TMM1);

            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM4, TMM7>();

            _tile_stored(TMM0, &a[0][cur_k_pos], a_s);
            _tile_stored(TMM1, &a[0][cur_k_pos_r], a_s);
        }

        cur_k_pos += k_block * 2;

        _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);

        _tile_zero(TMM0);

        __tile_dpbssd<TMM0, TMM4, TMM6>();

        _tile_stored(TMM0, &a[0][cur_k_pos], a_s);
        // add softmax
        softmax_tail.ref(apro_buffer, a_buffer, &att_mask, M, oscale);
        mha_init_av_tile(&tilecfg, rt_v);
        av_gemm_kernel<16>(apro_buffer, v_buffer, &r[cur_q_pos][0], &c[cur_q_pos][0], sl_pad, 0, step, M2);
        mha_init_qk_tile(&tilecfg);
    }
};

status_t amx_qk_gemm_softmax(const void* q_ptr, const void* k_ptr, 
                             void* a_buffer, void* v_buffer, void* apro_buffer, void* r_buffer, void* a_ptr, 
                             int nbq_row, int ldq, int q_block, int k_block, int rollback, 
                             float M, float oscale, int32_t att_mask, float M2) {
    /*
    do single qk gemm
    */

    switch (nbq_row) {
    case (1) :
        amx_qk_gemm_softmax_impl<0, 0>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (2) :
        amx_qk_gemm_softmax_impl<1, 1>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (3) :
        amx_qk_gemm_softmax_impl<0, 1>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (4) :
        amx_qk_gemm_softmax_impl<1, 2>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (5) :
        amx_qk_gemm_softmax_impl<0, 2>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (6) :
        amx_qk_gemm_softmax_impl<1, 3>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (7) :
        amx_qk_gemm_softmax_impl<0, 3>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (8) :
        amx_qk_gemm_softmax_impl<1, 4>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (9) :
        amx_qk_gemm_softmax_impl<0, 4>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (10) :
        amx_qk_gemm_softmax_impl<1, 5>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (11) :
        amx_qk_gemm_softmax_impl<0, 5>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (12) :
        amx_qk_gemm_softmax_impl<1, 6>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (13) :
        amx_qk_gemm_softmax_impl<0, 6>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (14) :
        amx_qk_gemm_softmax_impl<1, 7>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (15) :
        amx_qk_gemm_softmax_impl<0, 7>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (16) :
        amx_qk_gemm_softmax_impl<1, 8>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (17) :
        amx_qk_gemm_softmax_impl<0, 8>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (18) :
        amx_qk_gemm_softmax_impl<1, 9>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (19) :
        amx_qk_gemm_softmax_impl<0, 9>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (20) :
        amx_qk_gemm_softmax_impl<1, 10>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (21) :
        amx_qk_gemm_softmax_impl<0, 10>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (22) :
        amx_qk_gemm_softmax_impl<1, 11>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (23) :
        amx_qk_gemm_softmax_impl<0, 11>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    case (24) :
        amx_qk_gemm_softmax_impl<1, 12>::run(q_ptr, k_ptr, a_buffer, v_buffer, apro_buffer, r_buffer, 
                                            a_ptr, ldq, q_block, k_block, rollback, M, oscale, att_mask, M2);
        return status_t::success;
    }
    return status_t::failed;
}

at::Tensor amx_mha(
    const at::Tensor& qkv,
    const at::Tensor& att_mask,
    const at::Scalar& m1,
    const at::Scalar& oscale,
    const at::Scalar& m2
) {
    // std::cout << "call amx_mha" << std::endl;
    auto qkv_sizes = qkv.sizes();
    assert(qkv_sizes.size() == 3);
    auto bs = qkv_sizes[0];
    auto sl = qkv_sizes[1];
    auto stride = qkv_sizes[2];
    std::vector<int64_t> strides {sl * stride, stride, 1};
    auto qkv_block = stride / 3;
    int head_size = 64;
    int head_num = qkv_block / head_size;

    MHA_desc mhad(bs, sl, stride, head_num, head_size);
    mhad.init();

    auto origin_ptr = reinterpret_cast<int8_t (*)[sl][stride]>(qkv.data_ptr());
    auto att_mask_p = reinterpret_cast<int32_t *>(att_mask.data_ptr());
    
    auto amx_status = amx_init();
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
    int sl_pad = max_tile_row * mhad.nbq_row;
    auto attention = at::empty({bs, head_num, sl, head_size}, at::TensorOptions().
                        dtype<int8_t>().memory_format(c10::MemoryFormat::Contiguous));

    auto att_ptr = reinterpret_cast<int8_t (*)[head_num][sl][head_size]>(attention.data_ptr());

    // Dynamic allocate k_buffer
    int8_t k_buffer[sl_pad*64];
    int att_buffer[sl_pad*2*max_tile_row];
    uint8_t att_pro_buffer[sl_pad*2*max_tile_row];
    int8_t v_buffer[sl_pad*64];
    int r_buffer[sl*64];

    // only for test
    auto attention_int32 = at::empty({bs, head_num, sl, head_size}, at::TensorOptions().
                        dtype<int32_t>().memory_format(c10::MemoryFormat::Contiguous));

    auto att_ptr_int32 = reinterpret_cast<int32_t (*)[head_num][sl][head_size]>(attention_int32.data_ptr());

        
    for (int i = 0; i < bs; i++) // batch size
    {
        for (int j = 0; j < head_num; j++) // head num
        {
            // amx_status = amx_init();
            auto cur_q_ptr = &origin_ptr[i][0][j*head_size];
            auto cur_k_ptr = &origin_ptr[i][0][j*head_size+qkv_block];
            auto cur_v_ptr = &origin_ptr[i][0][j*head_size+qkv_block+qkv_block];
            auto cur_a_ptr = &att_ptr[i][j][0][0];
            auto cur_a_ptr_int32 = &att_ptr_int32[i][j][0][0];
            reorder_k_to_buffer_v2(cur_k_ptr, cur_v_ptr, 
                                   k_buffer, v_buffer, 
                                   sl, sl_pad, head_size, mhad.qkv_stride_);

            // amx_qk_gemm_softmax(cur_q_ptr, k_buffer, att_buffer, v_buffer, att_pro_buffer, r_buffer, cur_a_ptr, mhad.nbq_row, stride,
            //                     mhad.q_block, mhad.k_block, mhad.rollback, 
            //                     m1.toFloat(), oscale.toFloat(), att_mask_p[i], m2.toFloat());

            amx_qk_gemm_softmax(cur_q_ptr, k_buffer, att_buffer, v_buffer, att_pro_buffer, cur_a_ptr_int32, cur_a_ptr, mhad.nbq_row, stride,
                                mhad.q_block, mhad.k_block, mhad.rollback, 
                                m1.toFloat(), oscale.toFloat(), att_mask_p[i], m2.toFloat());
        }
    }

    return attention_int32;
}

}