#pragma once
#include <cstdlib>
#include <immintrin.h>
#include "sigmoid_tpp.hpp"
#include "tanh_tpp.hpp"
#include "el_common_intrin.hpp"
namespace intel_mlperf {

class lstm_postop_tpp{
public:
    static void ref(void* out_yt_bf16, void *out_yt_q, void *out_ht_q, void *it, void *ft, void *gt, void *ot, void *ct_1, float in_scale, float out_scale, int64_t line, bool last_layer_flag);
};

}
