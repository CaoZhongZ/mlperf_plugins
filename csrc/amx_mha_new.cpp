#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>
#include <string.h>
#include <iostream>

#include "amx_mha.hpp"
#include "amx_tdpbssd.hpp"
#include "i_softmax_tpp.hpp"
#include "helper.hpp"

namespace intel_mlperf {

status_t mha_init_tile(struct tilecfg *cfg, MHA_desc& mhad) {
    const int tile_config_size_in_bytes = 64;

    char* _tc = (char*)(cfg);
    for (int i = 0; i < tile_config_size_in_bytes; i++)
        _tc[i] = 0;

    int Qc = mhad.q_colsb;

    int Kc = mhad.k_colsb;

    for (int i = 0; i < mhad.nq_block; i++) {
        int Qr = (mhad.is_q_tail && i == mhad.nq_block - 1) ? mhad.q_tail : mhad.q_block;
        configure_tile(cfg, mhad.get_q_ntile(i), Qr, Qc);
    }

    for (int j = 0; j < mhad.nk_block; j++) {
        int Kr = mhad.k_block;
        configure_tile(cfg, mhad.get_k_ntile(j), Kr, Kc);
    }

    for (int i = 0; i < mhad.nq_block; i++) {
        int Ar = (mhad.is_q_tail && i == mhad.nq_block - 1) ? mhad.q_tail : mhad.q_block;
        for (int j = 0; j < mhad.nk_block; j++) {
            int Ac = mhad.k_block * mhad.typesize_A;
            configure_tile(cfg, mhad.get_a_ntile(i, j), Ar, Ac);
        }
    }

    cfg->palette = 1;
    _tile_release();
    _tile_loadconfig(cfg);

    return status_t::success;
}

status_t mha_init_qk_tile(struct tilecfg *cfg) {
    // all tile config to 16x64

    const int tile_config_size_in_bytes = 64;
    const int tile_num = 8;

    char* _tc = (char*)(cfg);
    for (int i = 0; i < tile_config_size_in_bytes; i++)
        _tc[i] = 0;

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
    // print_int8_2Dmatrix(k_buffer, 16, 16, row*4);
    return status_t::success;
}

status_t reorder_k_to_buffer_v2(const int8_t* k_ptr, int8_t* k_buffer, int row, int row_pad, int col, int stride)
{
    /// reorder k to k_buffer
    auto k_ptr_ = reinterpret_cast<const int (*)[stride/4]>(k_ptr);
    auto k_buffer_ = reinterpret_cast<int (*)[row_pad]>(k_buffer);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < row_pad; ++j) {
            k_buffer_[i][j] = j >= row ? 0 : k_ptr_[j][i];
        }
    }
    return status_t::success;
}

template <int even, int NBlock>
struct amx_qk_gemm_impl {
    inline static void run (const void* q_ptr, const void* k_ptr, void* a_ptr, int ldq, int q_block, int k_block, int rollback);
};

template <int NBlock>
struct amx_qk_gemm_impl<1, NBlock> {
    inline static void run (const void* q_ptr, const void* k_ptr, void* a_ptr, int ldq, int q_block, int k_block, int rollback) {
        // sl / 16 is even

        int sl_pad = NBlock * 32;
        auto q = reinterpret_cast<const int (*)[ldq/4]>(q_ptr);
        auto k = reinterpret_cast<const int (*)[sl_pad]>(k_ptr);
        auto a = reinterpret_cast<int (*)[sl_pad]>(a_ptr);

        int q_s = ldq;
        int k_s = sl_pad * 4;
        int a_s = k_s;

        int a_r_block = q_block;
        int a_c_block = k_block;

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
                // coordinate of A
                int cur_a_pos[2] = {i * a_r_block * 2, j * a_c_block * 2};
                int cur_a_pos_r[2] = {i * a_r_block * 2, j * a_c_block * 2 + a_c_block};
                int cur_a_pos_b[2] = {i * a_r_block * 2 + a_r_block, j * a_c_block * 2};
                int cur_a_pos_rb[2] = {i * a_r_block * 2 + a_r_block, j * a_c_block * 2 + a_c_block};

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

                _tile_stored(TMM0, &a[cur_a_pos[0]][cur_a_pos[1]], a_s);
                _tile_stored(TMM1, &a[cur_a_pos_r[0]][cur_a_pos_r[1]], a_s);
                _tile_stored(TMM2, &a[cur_a_pos_b[0]][cur_a_pos_b[1]], a_s);
                _tile_stored(TMM3, &a[cur_a_pos_rb[0]][cur_a_pos_rb[1]], a_s);
            }
        }
    }
};

template<int NBlock>
struct amx_qk_gemm_impl<0, NBlock> {
    inline static void run (const void* q_ptr, const void* k_ptr, void* a_ptr, int ldq, int q_block, int k_block, int rollback) {
        // sl / 16 is odd
        
        int sl_pad = NBlock * 32 + 16;
    
        auto q = reinterpret_cast<const int (*)[ldq/4]>(q_ptr);
        auto k = reinterpret_cast<const int (*)[sl_pad]>(k_ptr);
        auto a = reinterpret_cast<int (*)[sl_pad]>(a_ptr);

        int q_s = ldq;
        int k_s = sl_pad * 4;
        int a_s = k_s;

        int a_r_block = q_block;
        int a_c_block = k_block;

        int cur_q_pos = 0;
        int cur_q_pos_b = 0;
        int cur_k_pos = 0;
        int cur_k_pos_r = 0;
        int cur_a_pos[2] = {0, 0};
        int cur_a_pos_r[2] = {0, 0};
        int cur_a_pos_b[2] = {0, 0};
        int cur_a_pos_rb[2] = {0, 0};

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
                cur_a_pos[0] = i * a_r_block * 2;
                cur_a_pos[1] = j * a_c_block * 2;
                cur_a_pos_r[0] = i * a_r_block * 2;
                cur_a_pos_r[1] = j * a_c_block * 2 + a_c_block;
                cur_a_pos_b[0] = i * a_r_block * 2 + a_r_block;
                cur_a_pos_b[1] = j * a_c_block * 2;
                cur_a_pos_rb[0] = i * a_r_block * 2 + a_r_block;
                cur_a_pos_rb[1] = j * a_c_block * 2 + a_c_block;

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

                _tile_stored(TMM0, &a[cur_a_pos[0]][cur_a_pos[1]], a_s);
                _tile_stored(TMM1, &a[cur_a_pos_r[0]][cur_a_pos_r[1]], a_s);
                _tile_stored(TMM2, &a[cur_a_pos_b[0]][cur_a_pos_b[1]], a_s);
                _tile_stored(TMM3, &a[cur_a_pos_rb[0]][cur_a_pos_rb[1]], a_s);
            }

            cur_k_pos += k_block * 2;
            cur_a_pos[1] += a_c_block * 2;
            cur_a_pos_b[1] += a_c_block * 2;

            _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);

            _tile_zero(TMM0);
            _tile_zero(TMM1);
            
            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM5, TMM6>();

            _tile_stored(TMM0, &a[cur_a_pos[0]][cur_a_pos[1]], a_s);
            _tile_stored(TMM1, &a[cur_a_pos_b[0]][cur_a_pos_b[1]], a_s);
        }
        // the last 16 tile
        cur_q_pos += q_block - rollback;
        cur_a_pos[0] += a_r_block - rollback;
        cur_a_pos_r[0] += a_r_block - rollback;
        _tile_loadd(TMM4, &q[cur_q_pos][0], q_s);

#       pragma unroll NBlock
        for (int j = 0; j < NBlock; j++) {
            cur_k_pos = j * k_block * 2;
            cur_k_pos_r = cur_k_pos + k_block;

            cur_a_pos[1] = j * a_c_block * 2;
            cur_a_pos_r[1] = j * a_c_block * 2 + a_c_block;

            _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);
            _tile_loadd(TMM7, &k[0][cur_k_pos_r], k_s);

            _tile_zero(TMM0);
            _tile_zero(TMM1);

            __tile_dpbssd<TMM0, TMM4, TMM6>();
            __tile_dpbssd<TMM1, TMM4, TMM7>();

            _tile_stored(TMM0, &a[cur_a_pos[0]][cur_a_pos[1]], a_s);
            _tile_stored(TMM1, &a[cur_a_pos_r[0]][cur_a_pos_r[1]], a_s);
        }

        cur_k_pos += k_block * 2;
        cur_a_pos[1] += a_c_block * 2;

        _tile_loadd(TMM6, &k[0][cur_k_pos], k_s);

        _tile_zero(TMM0);

        __tile_dpbssd<TMM0, TMM4, TMM6>();

        _tile_stored(TMM0, &a[cur_a_pos[0]][cur_a_pos[1]], a_s);
    }
};

status_t amx_qk_gemm(const int8_t* q_ptr, const int8_t* k_ptr, int* a_ptr, MHA_desc& mhad) {
    /*
    do single qk gemm
    */
    
    int q_block = mhad.q_block;
    int k_block = mhad.k_block;
    int rollback = mhad.q_block - mhad.q_tail;

    switch (mhad.nbq_row) {
    case (1) :
        amx_qk_gemm_impl<0, 1>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (2) :
        amx_qk_gemm_impl<1, 1>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (3) :
        amx_qk_gemm_impl<0, 2>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (4) :
        amx_qk_gemm_impl<1, 2>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (5) :
        amx_qk_gemm_impl<0, 3>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (6) :
        amx_qk_gemm_impl<1, 3>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (7) :
        amx_qk_gemm_impl<0, 4>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (8) :
        amx_qk_gemm_impl<1, 4>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (9) :
        amx_qk_gemm_impl<0, 5>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (10) :
        amx_qk_gemm_impl<1, 5>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (11) :
        amx_qk_gemm_impl<0, 6>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (12) :
        amx_qk_gemm_impl<1, 6>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (13) :
        amx_qk_gemm_impl<0, 7>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (14) :
        amx_qk_gemm_impl<1, 7>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (15) :
        amx_qk_gemm_impl<0, 8>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (16) :
        amx_qk_gemm_impl<1, 8>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (17) :
        amx_qk_gemm_impl<0, 9>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (18) :
        amx_qk_gemm_impl<1, 9>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (19) :
        amx_qk_gemm_impl<0, 10>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (20) :
        amx_qk_gemm_impl<1, 10>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (21) :
        amx_qk_gemm_impl<0, 11>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (22) :
        amx_qk_gemm_impl<1, 11>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (23) :
        amx_qk_gemm_impl<0, 12>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    case (24) :
        amx_qk_gemm_impl<1, 12>::run(q_ptr, k_ptr, a_ptr, mhad.qkv_stride_, q_block, k_block, rollback);
        return status_t::success;
    }
    return status_t::failed;
}

at::Tensor amx_mha(
    const at::Tensor& qkv,
    const at::Tensor& att_mask,
    const at::Scalar& m1,
    const at::Scalar& oscale 
) {
    std::cout << "call amx_mha" << std::endl;
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
    
    auto amx_status = amx_init();
    if (!amx_status) {
        printf("amx init failed!\n");
        return qkv;
    }
    // mha_init_tile(&tilecfg, mhad);
    mha_init_qk_tile(&tilecfg);

    // create attention tensor
    int sl_pad = max_tile_row * mhad.nbq_row;
    auto attention = at::empty({bs, head_num, sl, sl_pad}, at::TensorOptions().
                        dtype<int>().memory_format(c10::MemoryFormat::Contiguous));

    auto att_ptr = reinterpret_cast<int (*)[head_num][sl][sl_pad]>(attention.data_ptr());
    std::vector<int64_t> att_strides = {head_num*sl*sl_pad, sl*sl_pad, sl_pad, 1};

    // Dynamic allocate k_buffer
    int8_t k_buffer[sl_pad*64];
    int att_buffer[sl_pad*2*max_tile_row];
    
    // do amx gemm
    for (int i = 0; i < bs; i++) // batch size
    {
        for (int j = 0; j < head_num; j++) // head num
        {
            auto cur_q_ptr = &origin_ptr[i][0][j*head_size];
            auto cur_k_ptr = &origin_ptr[i][0][j*head_size+qkv_block];
            auto cur_a_ptr = &att_ptr[i][j][0][0];
            reorder_k_to_buffer_v2(cur_k_ptr, k_buffer, sl, sl_pad, head_size, mhad.qkv_stride_);

            amx_qk_gemm(cur_q_ptr, k_buffer, cur_a_ptr, mhad);
        }
    }

    // add softmax
    // auto atten_size = attention.sizes();
    // i_softmax_tpp<16> softmax_compute(atten_size[0], atten_size[1], atten_size[2], atten_size[3]);
    // auto atten_probs = at::empty(atten_size, 
    //     at::TensorOptions().dtype<int8_t>()
    //     .memory_format(c10::MemoryFormat::Contiguous));

    // auto att_sz = att_mask.sizes();
    // auto* patt = reinterpret_cast<int32_t *>(att_mask.data_ptr());

    // softmax_compute.ref(
    //     atten_probs.data_ptr(), attention.data_ptr(),
    //     patt, m1.toFloat(), oscale.toFloat());

    return attention;
}

}