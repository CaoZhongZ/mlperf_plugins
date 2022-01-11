#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>
#include <iostream>

#include "amx_mha.hpp"
#include "amx_tdpbssd.hpp"

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

    if (row * col > 16 * MAX_SL * 4) {
        return status_t::failed;
    }

    // auto ptr = k_ptr;
    int ks = MAX_SL * 4;
    int nc_block = col / 4;
    for (int i = 0; i < row; i++)
    {
        int kc = i * 4;
        for (int j = 0; j < col; j += 4)
        {
            int kr = j / 4;
            
            k_buffer[kr*ks+kc] = k_ptr[i*stride+j];
            k_buffer[kr*ks+kc+1] = k_ptr[i*stride+j+1];
            k_buffer[kr*ks+kc+2] = k_ptr[i*stride+j+2];
            k_buffer[kr*ks+kc+3] = k_ptr[i*stride+j+3];
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
    auto qkv_block = stride / 3;
    int head_size = 64;

    int8_t* origin_ptr = (int8_t*)qkv.data_ptr();
    auto k_ptr = origin_ptr + qkv_block;

    reorder_k_to_buffer(k_ptr, sl, head_size, stride);


    auto options = torch::TensorOptions().dtype(torch::kInt8);
    auto k_buffer_tensor = torch::from_blob((void*)k_buffer, {16, sl*4}, {MAX_SL*4, 1}, options);

    std::cout << std::endl;
    return k_buffer_tensor;
}

}