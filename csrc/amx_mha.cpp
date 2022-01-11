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
    auto ld = stride / 3;
    int head_size = 64;

    int8_t* origin_ptr = (int8_t*)qkv.data_ptr();
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            std::cout << static_cast<int>(origin_ptr[j]) << "\t";
        }
        std::cout << std::endl;
        origin_ptr += stride;
        
    }
    std::cout << std::endl;
    return qkv;
}

}