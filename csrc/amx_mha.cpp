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
    constexpr int tile_config_size_in_bytes = 64;

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

status_t amx_qk_gemm(const int8_t* q_ptr, const int8_t* k_ptr, int* a_ptr, MHA_desc& mhad) {
    /*
    do single qk gemm
    */
    
    int q_stride = mhad.qkv_stride_ / 4;
    
    auto q = reinterpret_cast<const int (*)[q_stride]>(q_ptr);
    auto k = reinterpret_cast<const int (*)[mhad.sl_pad]>(k_ptr);
    auto a = reinterpret_cast<int (*)[mhad.sl_pad]>(a_ptr);

    if (mhad.is_q_tail) {
        for (int i = 0; i < mhad.nbq_row; i++) {
            bool cur_q_tail = i == mhad.nbq_row - 1;
            auto cur_q_ptr = &q[i*mhad.q_block][0];
            if (cur_q_tail) {
                _tile_loadd(TMM5, cur_q_ptr, mhad.qkv_stride_);
            } else {
                _tile_loadd(TMM4, cur_q_ptr, mhad.qkv_stride_);
            }
            
            for (int j = 0; j < mhad.nbk_col; j += mhad.nk_block) {
                // double tiles go
                auto cur_k_ptr = &k[0][j*mhad.k_block];
                auto cur_a_ptr = &a[i*mhad.a_r_block][j*mhad.a_c_block];
                if (j == mhad.nbk_col - 1) {
                    _tile_loadd(TMM6, cur_k_ptr, mhad.sl_pad*4);
                    if (cur_q_tail) {
                        _tile_zero(TMM2);
                        __tile_dpbssd<TMM2, TMM5, TMM6>();
                        // TODO: calculate pre
                        _tile_stored(TMM2, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                    } else {
                        _tile_zero(TMM0);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                    }
                } else {
                    auto cur_k_ptr_r = &k[0][(j+1)*mhad.k_block];
                    auto cur_a_ptr_r = &a[i*mhad.a_r_block][(j+1)*mhad.a_c_block];
                    _tile_loadd(TMM6, cur_k_ptr, mhad.sl_pad*4);
                    _tile_loadd(TMM7, cur_k_ptr_r, mhad.sl_pad*4);
                    if (cur_q_tail) {
                        _tile_zero(TMM2);
                        _tile_zero(TMM3);
                        __tile_dpbssd<TMM2, TMM5, TMM6>();
                        __tile_dpbssd<TMM3, TMM5, TMM7>();
                        _tile_stored(TMM2, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM3, cur_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                    } else {
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM4, TMM7>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM1, cur_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < mhad.nbq_row; i += mhad.nq_block) {
            auto cur_q_ptr = &q[i*mhad.q_block][0];
            if (i == mhad.nbq_row - 1) {
                _tile_loadd(TMM4, cur_q_ptr, mhad.qkv_stride_);
            } else {
                auto cur_q_ptr_b = &q[(i+1)*mhad.q_block][0];
                _tile_loadd(TMM4, cur_q_ptr, mhad.qkv_stride_);
                _tile_loadd(TMM5, cur_q_ptr_b, mhad.qkv_stride_);
            }
            for (int j = 0; j < mhad.nbk_col; j += mhad.nk_block) {
                auto cur_k_ptr = &k[0][j*mhad.k_block];
                auto cur_a_ptr = &a[i*mhad.a_r_block][j*mhad.a_c_block];
                if (j == mhad.nbk_col - 1) {
                    _tile_loadd(TMM6, cur_k_ptr, mhad.sl_pad*4);
                    if (i == mhad.nbq_row - 1) {
                        _tile_zero(TMM0);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                    } else {
                        auto cur_a_ptr_b = &a[(i+1)*mhad.a_r_block][j*mhad.a_c_block];
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM5, TMM6>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM1, cur_a_ptr_b, mhad.att_stride_ * mhad.typesize_A);
                    }
                } else {
                    auto cur_k_ptr_r = &k[0][(j+1)*mhad.k_block];
                    _tile_loadd(TMM6, cur_k_ptr, mhad.sl_pad*4);
                    _tile_loadd(TMM7, cur_k_ptr_r, mhad.sl_pad*4);
                    if (i == mhad.nbq_row - 1) {
                        auto cur_a_ptr_r = &a[i*mhad.a_r_block][(j+1)*mhad.a_c_block];
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM4, TMM7>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM1, cur_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                    } else {
                        auto cur_a_ptr_r = &a[i*mhad.a_r_block][(j+1)*mhad.a_c_block];
                        auto cur_a_ptr_b = &a[(i+1)*mhad.a_r_block][j*mhad.a_c_block];
                        auto cur_a_ptr_rb = &a[(i+1)*mhad.a_r_block][(j+1)*mhad.a_c_block];
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        _tile_zero(TMM2);
                        _tile_zero(TMM3);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM4, TMM7>();
                        __tile_dpbssd<TMM2, TMM5, TMM6>();
                        __tile_dpbssd<TMM3, TMM5, TMM7>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM1, cur_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM2, cur_a_ptr_b, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM3, cur_a_ptr_rb, mhad.att_stride_ * mhad.typesize_A);
                    }
                }
            }
        }
    }
    
    return status_t::success;


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
    mha_init_tile(&tilecfg, mhad);

    // create attention tensor
    int sl_pad = max_tile_row * mhad.nbq_row;
    auto attention = at::empty({bs, head_num, sl, sl_pad}, at::TensorOptions().
                        dtype<int>().memory_format(c10::MemoryFormat::Contiguous));

    auto att_ptr = reinterpret_cast<int (*)[head_num][sl][sl_pad]>(attention.data_ptr());
    std::vector<int64_t> att_strides = {head_num*sl*sl_pad, sl*sl_pad, sl_pad, 1};

    // Dynamic allocate k_buffer
    int8_t k_buffer[sl_pad*64];
    
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
    auto atten_size = attention.sizes();
    i_softmax_tpp<16> softmax_compute(atten_size[0], atten_size[1], atten_size[2], atten_size[3]);
    auto atten_probs = at::empty(atten_size, 
        at::TensorOptions().dtype<int8_t>()
        .memory_format(c10::MemoryFormat::Contiguous));

    auto att_sz = att_mask.sizes();
    auto* patt = reinterpret_cast<int32_t *>(att_mask.data_ptr());

    softmax_compute.ref(
        atten_probs.data_ptr(), attention.data_ptr(),
        patt, m1.toFloat(), oscale.toFloat());

    return atten_probs;
}

}