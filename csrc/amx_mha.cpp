#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>
#include <string.h>
#include <iostream>

#include "amx_mha.hpp"
#include "amx_tdpbssd.hpp"
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

status_t reorder_k_to_buffer(const int8_t* k_ptr, int row, int col, int stride) {
    /*
    reorder k from sl*64 to 16*sl*4
    */

    if (row * col > 16 * max_sl * 4) {
        return status_t::failed;
    }

    // auto ptr = k_ptr;
    int ks = max_sl * 4;
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
    return status_t::success;
}

status_t amx_qk_gemm(const int8_t* q_ptr, const int8_t* k_ptr, int* a_ptr, MHA_desc& mhad) {
    /*
    do single qk gemm
    */

    
    if (mhad.is_q_tail) {
        for (int i = 0; i < mhad.nbq_row; i++) {
            bool cur_q_tail = i == mhad.nbq_row - 1;
            auto cur_q_ptr = q_ptr + i * mhad.q_block * mhad.qkv_stride_;
            if (cur_q_tail) {
                _tile_loadd(TMM5, cur_q_ptr, mhad.qkv_stride_);
            } else {
                _tile_loadd(TMM4, cur_q_ptr, mhad.qkv_stride_);
            }
            
            for (int j = 0; j < mhad.nbk_col; j += mhad.nk_block) {
                // double tiles go
                auto cur_k_ptr = k_ptr + j * mhad.k_colsb;
                auto cur_a_ptr = a_ptr + i * mhad.a_r_block * mhad.att_stride_ + j * mhad.a_c_block;
                if (j == mhad.nbk_col - 1) {
                    _tile_loadd(TMM6, cur_k_ptr, max_sl*4);
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
                    _tile_loadd(TMM6, cur_k_ptr, max_sl*4);
                    _tile_loadd(TMM7, cur_k_ptr+mhad.k_colsb, max_sl*4);
                    if (cur_q_tail) {
                        _tile_zero(TMM2);
                        _tile_zero(TMM3);
                        __tile_dpbssd<TMM2, TMM5, TMM6>();
                        __tile_dpbssd<TMM3, TMM5, TMM7>();
                        _tile_stored(TMM2, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        auto tmp_a_ptr_r = cur_a_ptr+mhad.a_c_block;
                        _tile_stored(TMM3, tmp_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                    } else {
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM4, TMM7>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        auto tmp_a_ptr_r = cur_a_ptr+mhad.a_c_block;
                        _tile_stored(TMM1, tmp_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < mhad.nbq_row; i += mhad.nq_block) {
            auto cur_q_ptr = q_ptr + i * mhad.q_block * mhad.qkv_stride_;
            if (i == mhad.nbq_row - 1) {
                _tile_loadd(TMM4, cur_q_ptr, mhad.qkv_stride_);
            } else {
                _tile_loadd(TMM4, cur_q_ptr, mhad.qkv_stride_);
                _tile_loadd(TMM5, cur_q_ptr+mhad.q_block*mhad.qkv_stride_, mhad.qkv_stride_);
            }
            for (int j = 0; j < mhad.nbk_col; j += mhad.nk_block) {
                auto cur_k_ptr = k_ptr + j * mhad.k_colsb;
                auto cur_a_ptr = a_ptr + i * mhad.a_r_block * mhad.att_stride_ + j * mhad.a_c_block;
                if (j == mhad.nbk_col - 1) {
                    _tile_loadd(TMM6, cur_k_ptr, max_sl*4);
                    if (i == mhad.nbq_row - 1) {
                        _tile_zero(TMM0);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                    } else {
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM5, TMM6>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        auto tmp_a_ptr_b = cur_a_ptr+mhad.a_r_block*mhad.att_stride_;
                        _tile_stored(TMM1, tmp_a_ptr_b, mhad.att_stride_ * mhad.typesize_A);
                    }
                } else {
                    _tile_loadd(TMM6, cur_k_ptr, max_sl*4);
                    _tile_loadd(TMM7, cur_k_ptr+mhad.k_colsb, max_sl*4);
                    if (i == mhad.nbq_row - 1) {
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM4, TMM7>();
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        auto tmp_a_ptr_r = cur_a_ptr+mhad.a_c_block;
                        _tile_stored(TMM1, tmp_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                    } else {
                        _tile_zero(TMM0);
                        _tile_zero(TMM1);
                        _tile_zero(TMM2);
                        _tile_zero(TMM3);
                        __tile_dpbssd<TMM0, TMM4, TMM6>();
                        __tile_dpbssd<TMM1, TMM4, TMM7>();
                        __tile_dpbssd<TMM2, TMM5, TMM6>();
                        __tile_dpbssd<TMM3, TMM5, TMM7>();
                        auto tmp_a_ptr_r = cur_a_ptr+mhad.a_c_block;
                        auto tmp_a_ptr_b = cur_a_ptr+mhad.a_r_block*mhad.att_stride_;
                        auto tmp_a_ptr_rb = cur_a_ptr+mhad.a_r_block*mhad.att_stride_+mhad.a_c_block;
                        _tile_stored(TMM0, cur_a_ptr, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM1, tmp_a_ptr_r, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM2, tmp_a_ptr_b, mhad.att_stride_ * mhad.typesize_A);
                        _tile_stored(TMM3, tmp_a_ptr_rb, mhad.att_stride_ * mhad.typesize_A);
                    }
                }
            }
        }
    }
    
    return status_t::success;


}

at::Tensor amx_mha(
    const at::Tensor& qkv,
    const at::Tensor& attpro,
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

    int8_t* origin_ptr = (int8_t*)qkv.data_ptr();
    // auto k_ptr = origin_ptr + qkv_block;
    
    /* test reorder k to buffer function */
    // reorder_k_to_buffer(k_ptr, sl, head_size, stride);

    
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

    auto att_ptr = (int*)attention.data_ptr();
    std::vector<int64_t> att_strides = {head_num*sl*sl_pad, sl*sl_pad, sl_pad, 1};

    // do amx gemm
    for (int i = 0; i < bs; i++) // batch size
    {
        for (int j = 0; j < head_num; j++) // head num
        {
            auto cur_q_ptr = origin_ptr + i * strides[0] + j * head_size;
            auto cur_k_ptr = cur_q_ptr + qkv_block;
            auto cur_a_ptr = att_ptr + i * att_strides[0] + j * att_strides[1];
            reorder_k_to_buffer(cur_k_ptr, sl, head_size, mhad.qkv_stride_);

            amx_qk_gemm(cur_q_ptr, k_buffer, cur_a_ptr, mhad);
        }
    }

    // auto options = torch::TensorOptions().dtype(torch::kInt8);
    // auto k_buffer_tensor = torch::from_blob((void*)k_buffer, {16, sl*4}, {max_sl*4, 1}, options);

    // std::cout << std::endl;
    return attention;
}

}