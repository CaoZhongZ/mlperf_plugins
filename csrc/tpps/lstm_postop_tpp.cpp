#include "lstm_postop_tpp.hpp"
#include <cstdlib>
#include <iostream>

namespace intel_mlperf {

void lstm_postop_tpp::ref(void *out_yt_q, void *out_ht_q, void *it, void *ft, void *gt, void *ot, void *ct_1, float in_scale, float out_scale, int64_t line, bool last_layer_flag){
    
    // compute four post-op
    auto pin_it = reinterpret_cast<float *>(it);
    auto pin_ft = reinterpret_cast<float *>(ft);
    auto pin_gt = reinterpret_cast<float *>(gt);
    auto pin_ot = reinterpret_cast<float *>(ot);

    alignas(64) _Float16 it_out[line];
    alignas(64) _Float16 ft_out[line];
    alignas(64) _Float16 gt_out[line];
    alignas(64) _Float16 ot_out[line];
    sigmoid_tpp<32>::ref(it_out, pin_it,line);
    sigmoid_tpp<32>::ref(ft_out, pin_ft,line);
    tanh_tpp<32>::ref(gt_out, pin_gt,line);
    sigmoid_tpp<32>::ref(ot_out, pin_ot,line);

    // comput ct
    int64_t half_len = 32;
    auto n_batch = line / half_len;
    auto pin_ct = reinterpret_cast<_Float16 *>(ct_1);
    #pragma unroll(32)
    for(int j=0;j<n_batch;j++){ 
        auto i = _mm512_loadu_ph((&it_out[j*32]));
        auto f = _mm512_loadu_ph((&ft_out[j*32]));
        auto g = _mm512_loadu_ph((&gt_out[j*32]));
        auto c = _mm512_loadu_ph((&pin_ct[j*32]));
        auto a = _mm512_mul_ph(f, c);
        auto o = _mm512_fmadd_ph(i, g, a);
        _mm512_store_ph(&pin_ct[j*32],o);
    }

    // inplace it_out and ft_out; it_out = ht_out(fp16), ft_out = ct_tanh
    tanh_tpp<32>::ref_(ft_out,pin_ct,line);
    #pragma unroll(32)
    for(int j=0,z=0;j<n_batch;j++,z=z+2){
        auto a = _mm512_loadu_ph(&ot_out[j*32]);
        auto b = _mm512_loadu_ph(&ft_out[j*32]);
        auto o_ht = _mm512_mul_ph(a,b);
        _mm512_store_ph(&it_out[j*32],o_ht);
        if(last_layer_flag){
            // ht:fp16->fp32
            auto y_1 = _mm512_extractf32x8_ps(o_ht,0);
            auto y_2 = _mm512_extractf32x8_ps(o_ht,1);
            auto o_1 = _mm512_cvtxph_ps(_mm256_castps_ph(y_1));
            auto o_2 = _mm512_cvtxph_ps(_mm256_castps_ph(y_2));
            _mm512_store_ps(&pin_it[z*16],o_1);
            _mm512_store_ps(&pin_it[(z+1)*16],o_2);
        }
    }

    // quant
    auto pout_yt_q = reinterpret_cast<int8_t *>(out_yt_q);
    
    if(!last_layer_flag){
        #pragma unroll(32)
        for(int j=0;j<n_batch;j++){
            auto vout_scale = _mm512_set1_ph(out_scale);
            auto ht_ph = _mm512_loadu_ph(&it_out[j*32]);
            auto yt_quant = _mm512_scale_min128max_i8_ph(ht_ph, vout_scale);
            _mm512_mask_cvtepi16_storeu_epi8(&pout_yt_q[j*32], 0xffffffff, yt_quant);
        }
    }

    auto pout_ht_q = reinterpret_cast<int8_t *>(out_ht_q);
    #pragma unroll(32)
    for(int j=0;j<n_batch;j++){
            auto vin_scale = _mm512_set1_ph(in_scale);
            auto ht_ph = _mm512_loadu_ph(&it_out[j*32]);
            auto ht_quant = _mm512_scale_min128max_i8_ph(ht_ph, vin_scale);
            _mm512_mask_cvtepi16_storeu_epi8(&pout_ht_q[j*32], 0xffffffff, ht_quant);
    }
    
    // auto pfp16 = _mm512_loadu_ph(&ct_out[992]);
    // // auto pfp16 = _mm512_loadu_ph(ht_out);
    // helper::_mm512_print_ph(pfp16);
    // std::cout << "aaa" << std::endl;
}

}