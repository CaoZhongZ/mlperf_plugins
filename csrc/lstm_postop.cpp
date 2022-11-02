#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include <vector>
#include "lstm_postop_tpp.hpp"

namespace intel_mlperf {

std::vector<at::Tensor> lstm_postop (
  const at::Tensor& it,
  const at::Tensor& ft,
  const at::Tensor& gt,
  const at::Tensor& ot,
  const at::Tensor& ct_1,
  const c10::optional<at::Scalar>& input_scale,
  const c10::optional<at::Scalar>& output_scale,
  const bool& last_layer_flag) {
    std::vector<at::Tensor> output = {};
    auto sizes = it.sizes();
    auto batch = sizes[0];
    auto line = sizes[1];
    auto stride = it.strides();
    auto lda = stride[0];
    auto in_scale = input_scale->toFloat();
    auto out_scale = output_scale->toFloat();

    auto output_1 = at::empty(sizes,
    at::TensorOptions().dtype<float>()
    .memory_format(c10::MemoryFormat::Contiguous));

    auto output_1_q = at::empty(sizes,
    at::TensorOptions().dtype<int8_t>()
    .memory_format(c10::MemoryFormat::Contiguous));

    auto output_2 = at::empty(sizes,
    at::TensorOptions().dtype<int8_t>()
    .memory_format(c10::MemoryFormat::Contiguous));

    auto output_3 = at::empty(sizes,
    at::TensorOptions().dtype<at::Half>()
    .memory_format(c10::MemoryFormat::Contiguous));

    auto *in_it = it.data_ptr();
    auto *in_ft = ft.data_ptr();
    auto *in_gt = gt.data_ptr();
    auto *in_ot = ot.data_ptr();
    auto *in_ct = ct_1.data_ptr();
    auto *out_1 = output_1.data_ptr();
    auto *out_1_q = output_1_q.data_ptr();
    auto *out_2 = output_2.data_ptr();
    auto *out_3 = output_3.data_ptr();

    // _Float16 it_out[line];

    #pragma omp parallel for
    for (auto b=0; b < batch;++b) {
      auto pin_it = reinterpret_cast<float (*)[line]>(in_it);
      auto pin_ft = reinterpret_cast<float (*)[line]>(in_ft);
      auto pin_gt = reinterpret_cast<float (*)[line]>(in_gt);
      auto pin_ot = reinterpret_cast<float (*)[line]>(in_ot);
      auto pin_ct = reinterpret_cast<_Float16 (*)[line]>(in_ct);
      // TODO 
      // if(last_layer_flag)
      //   auto pout_1 = reinterpret_cast<at::Half (*)[line]>(out_1);
      // else
      auto pout_1 = reinterpret_cast<float (*)[line]>(out_1);
      auto pout_1_q = reinterpret_cast<int8_t (*)[line]>(out_1_q);
      auto pout_2 = reinterpret_cast<int8_t (*)[line]>(out_2);
      auto pout_3 = reinterpret_cast<at::Half (*)[line]>(out_3);

      if(lda==line)
        lstm_postop_tpp::ref(pout_1[b],pout_1_q[b],pout_2[b],pout_3[b],pin_it[b],pin_ft[b],pin_gt[b],pin_ot[b],pin_ct[b],in_scale,out_scale,line,last_layer_flag);
      else
        lstm_postop_tpp::ref(pout_1[b],pout_1_q[b],pout_2[b],pout_3[b],pin_it[4*b],pin_ft[4*b],pin_gt[4*b],pin_ot[4*b],pin_ct[b],in_scale,out_scale,line,last_layer_flag);
    }

    output.push_back(output_1);
    output.push_back(output_1_q);
    output.push_back(output_2);
    output.push_back(output_3);

    return output;
}

}
