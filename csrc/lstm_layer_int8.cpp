#include <ATen/Functions.h>
#include <ATen/ops/add_ops.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <string.h>
#include <vector>

#include "amx_config.hpp"
#include "amx_init.hpp"
#include "lstm_layer_int8.hpp"
#include "tpps/i_linear_tpp.hpp"
#include "tpps/lstm_postop_tpp.hpp"

namespace intel_mlperf {

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_int8(
    const at::Tensor &x, const at::Tensor &hx, const at::Tensor &cx,
    const at::Tensor &w_ih, const at::Tensor &w_hh, const at::Tensor &b_ih,
    const at::Tensor &b_hh, const c10::optional<at::Scalar> &rb_scale,
    const c10::optional<at::Scalar> &i_scale,
    const c10::optional<at::Scalar> &o_scale, const bool skip_quant_y) {
  auto ishape = x.sizes();
  auto seq_len = ishape[0];
  auto bs = ishape[1];
  auto input_f = ishape[2];
  auto hidden = hx.size(1);
  auto output_f = b_ih.size(0);

  float rb_scale_ = rb_scale.value_or(at::Scalar(1.f)).toFloat();
  float i_scale_ = i_scale.value_or(at::Scalar(1.f)).toFloat();
  float o_scale_ = o_scale.value_or(at::Scalar(1.f)).toFloat();

  // output shape: [sl, batch_size, output_f]
  auto y = at::empty({seq_len, bs, output_f},
                     at::TensorOptions().dtype<float>().memory_format(
                         c10::MemoryFormat::Contiguous));
  auto y_q = at::empty({seq_len, bs, hidden},
                       at::TensorOptions().dtype<int8_t>().memory_format(
                           c10::MemoryFormat::Contiguous));
  auto linear_ih = i_linear_i8o32(bs, input_f, output_f, true, false);

  auto x_ptr = x.data_ptr();
  auto w_ih_ptr = w_ih.data_ptr();
  auto y_ptr = y.data_ptr();
  auto y_q_ptr = y_q.data_ptr();
  auto b_ih_ptr = b_ih.data_ptr();

  amx_init::amx_init();
  for (int i = 0; i < seq_len; i++) {
    // linear for input
#pragma omp parallel
    {
      auto input_ = reinterpret_cast<int8_t(*)[bs][input_f]>(x_ptr);
      auto weight_ = reinterpret_cast<int8_t *>(w_ih_ptr);
      auto output_ = reinterpret_cast<float(*)[bs][output_f]>(y_ptr);
      auto bias_ = reinterpret_cast<float *>(b_ih_ptr);
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      linear_ih.tile_dot_product_16x256_shortage(output_[i], input_[i], weight_,
                                                 bias_, rb_scale_, 0.0, bs,
                                                 core_id, total_core_num);
      Tilecfg().release_config();
    }
  }

  auto y_i = torch::chunk(y, seq_len, 0);
  auto hy = at::empty({bs, output_f},
                      at::TensorOptions().dtype<float>().memory_format(
                          c10::MemoryFormat::Contiguous));
  auto linear_hh = i_linear_i8o32(bs, hidden, output_f, true, false);

  auto hx_ptr = hx.data_ptr();
  auto w_hh_ptr = w_hh.data_ptr();
  auto hy_ptr = hy.data_ptr();
  auto b_hh_ptr = b_hh.data_ptr();
  auto cx_ptr = cx.data_ptr();

  for (int i = 0; i < seq_len; i++) {
    // linear for hidden state
#pragma omp parallel
    {
      auto input_ = reinterpret_cast<int8_t *>(hx_ptr);
      auto weight_ = reinterpret_cast<int8_t *>(w_hh_ptr);
      auto output_ = reinterpret_cast<float *>(hy_ptr);
      auto bias_ = reinterpret_cast<float *>(b_hh_ptr);
      Tilecfg().set_config();
      auto total_core_num = omp_get_num_threads();
      auto core_id = omp_get_thread_num();
      linear_hh.tile_dot_product_16x256_shortage(output_, input_, weight_,
                                                 bias_, rb_scale_, 0.0, bs,
                                                 core_id, total_core_num);
      Tilecfg().release_config();
    }

    // y = hy + y
    at::_ops::add__Tensor::call(y_i[i], hy, 1.0);

    // post op
#pragma omp parallel for
    for (auto b = 0; b < bs; ++b) {
      auto y_ = reinterpret_cast<float(*)[bs][output_f]>(y_ptr);
      auto y_q_ = reinterpret_cast<int8_t(*)[bs][hidden]>(y_q_ptr);
      auto hx_ = reinterpret_cast<int8_t(*)[hidden]>(hx_ptr);
      auto cx_ = reinterpret_cast<_Float16(*)[hidden]>(cx_ptr);
      lstm_postop_tpp::ref(y_q_[i][b], hx_[b], &y_[i][b][0], &y_[i][b][hidden],
                           &y_[i][b][hidden * 2], &y_[i][b][hidden * 3], cx_[b],
                           i_scale_, o_scale_, hidden, skip_quant_y);
    }
  }
  if (skip_quant_y) {
    return {torch::chunk(y, 4, 2)[0], hx, cx};
  } else {
    return {y_q, hx, cx};
  }
}

} // namespace intel_mlperf
