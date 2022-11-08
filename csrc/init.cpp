#include <torch/library.h>
#include <plugins.hpp>

#include "amx_mha.hpp"
#include "amx_mha_concat.hpp"
#include "amx_linear.hpp"
#include "linear.hpp"
#include "softmax.hpp"
#include "activation.hpp"
#include "normalization.hpp"
#include "preemphasis.hpp"
#include "frame_splicing.hpp"
#include "lstm.hpp"
#include "lstm_postop.hpp"
#include "lstm_int8.hpp"

TORCH_LIBRARY(intel_mlperf, m) {
  m.def(
    "amx_mha(Tensor qkv, Tensor att_mask, Scalar m1, Scalar oscale, Scalar m2) -> Tensor",
    intel_mlperf::amx_mha);
  m.def(
    "amx_mha_concat(Tensor qkv, Tensor att_mask, Tensor length_ids, Scalar m1, Scalar oscale, Scalar m2) -> Tensor",
    intel_mlperf::amx_mha_concat);
  m.def(
    "amx_linear(Tensor input, Tensor weight, Tensor bias, Scalar scale, bool post_op, Scalar o_scale) -> Tensor",
    intel_mlperf::amx_linear);
  m.def(
    "amx_linear_i8o32(Tensor input, Tensor weight, Tensor bias, Scalar scale) -> Tensor",
    intel_mlperf::amx_linear_i8o32);
  m.def(
    "linear(Tensor input, Tensor weight, Tensor ? bias, Scalar ? scale, Scalar ? zero) -> Tensor",
    intel_mlperf::linear);
  m.def(
    "linear_gelu(Tensor input, Tensor weight, Tensor ? bias, Scalar ? M, Scalar ? scale, Scalar ? zero) -> Tensor",
    intel_mlperf::linear_gelu);
  m.def(
      "prepack_linear_weight(Tensor weight) -> Tensor",
      intel_mlperf::prepack_linear_weight);
  m.def("baddbmm_out_", intel_mlperf::baddbmm_out_);
  m.def("matmul_out_", intel_mlperf::matmul_out_);
  m.def("reorder_test(Tensor input) -> Tensor", intel_mlperf::reorder_test);
  m.def(
      "i_softmax(Tensor input, Tensor att_mask, Scalar M, Scalar oscale) -> Tensor",
      intel_mlperf::i_softmax
      );
  m.def(
      "i_softmax_u(Tensor input, Tensor att_mask, Scalar M, Scalar oscale) -> Tensor",
      intel_mlperf::i_softmax_u
      );
  m.def(
      "i_softmax_(Tensor(a!) input, Tensor att_mask, Scalar M, Scalar oscale) -> Tensor(a!)",
      intel_mlperf::i_softmax_
      );
  m.def(
      "i_gelu(Tensor input, Scalar M, Scalar oscale, Scalar o_off) -> Tensor",
      intel_mlperf::i_gelu);
  m.def(
      "i_identity(Tensor input, Scalar ? M, Scalar oscale, Scalar o_off) -> Tensor",
      intel_mlperf::i_identity);
  m.def(
      "i_identity_cin(Tensor input, Scalar oscale) -> Tensor",
      intel_mlperf::i_identity_cin);
  m.def(
      "i_identity_(Tensor(a!) input, Scalar ? M, Scalar oscale, Scalar o_off) -> Tensor(a!)",
      intel_mlperf::i_identity_);
  m.def(
      "i_residual_layernorm(Tensor input1, Tensor input2, Tensor weight, Tensor bias, Scalar scale_1, Scalar scale_2, Scalar oscale, Scalar ? eps, Scalar ? o_off) -> Tensor",
      intel_mlperf::i_residual_layernorm);
  m.def(
      "i_residual_layernorm_(Tensor(a!) input1, Tensor input2, Tensor weight, Tensor bias, Scalar scale_1, Scalar scale_2, Scalar oscale, Scalar ? eps, Scalar ? o_off) -> Tensor(a!)",
      intel_mlperf::i_residual_layernorm_);
  m.def(
      "i_residual_layernorm_cin_(Tensor(a!) input1, Tensor input2, Tensor weight, Tensor bias, Scalar scale_1, Scalar scale_2, Scalar oscale, Scalar ? eps, Scalar ? o_off) -> Tensor(a!)",
      intel_mlperf::i_residual_layernorm_cin_);
  m.def(
      "i_layernorm(Tensor input, Tensor weight, Tensor bias, Scalar oscale, Scalar ? eps, Scalar ? o_off) -> Tensor",
      intel_mlperf::i_layernorm);
  m.def(
      "i_layernorm_unpad(Tensor input, Tensor weight, Tensor bias, Tensor length, Scalar ? eps, Scalar ? unbiased) -> Tensor",
      intel_mlperf::i_layernorm_unpad);
  m.def(
      "preemphasis(Tensor input, Scalar ? coeff) -> Tensor",
      intel_mlperf::preemphasis);
  m.def(
      "frame_splicing(Tensor input, Scalar factor) -> Tensor",
      intel_mlperf::frame_splicing);
  m.def(
      "lstm(Tensor x, Tensor ? hx, Tensor ? cx, Tensor[][] all_weights, int ? hidden_size) -> (Tensor, Tensor, Tensor)",
      intel_mlperf::lstm);
  m.def(
      "lstm_layer(Tensor x, Tensor ? hx, Tensor ? cx, Tensor w_ih, Tensor w_hh, Tensor ? b_ih, Tensor ? b_hh, int ? hidden_size) -> (Tensor, Tensor, Tensor)",
      intel_mlperf::lstm_layer);
  m.def(
      "prepack_lstm_weights(Tensor w_ih, Tensor w_hh) -> (Tensor, Tensor)",
      intel_mlperf::prepack_lstm_weights);
  m.def(
      "tanh(Tensor _0) -> Tensor",
      intel_mlperf::tanh);
  m.def(
      "sigmoid(Tensor _0) -> Tensor",
      intel_mlperf::sigmoid);
  m.def(
      "tanh_f16(Tensor _0) -> Tensor",
      intel_mlperf::tanh_f16);
  m.def(
      "lstm_postop(Tensor _0, Tensor _1, Tensor _2, Tensor _3, Tensor _4, Scalar? _5, Scalar? _6, bool _7) -> Tensor[]",
      intel_mlperf::lstm_postop);
  m.def(
      "lstm_layer_int8(Tensor x, Tensor hx, Tensor cx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Scalar? rb_scale, Scalar? i_scale, Scalar? o_scale, bool skip_quant_y) -> (Tensor, Tensor, Tensor)",
      intel_mlperf::lstm_layer_int8);
  m.def(
      "lstm_int8(Tensor _0, Tensor[] _1, Tensor[] _2, Tensor[][] _3, Tensor _4, Tensor _5, Tensor _6, bool _7) -> (Tensor, Tensor[], Tensor[])",
      intel_mlperf::lstm_int8);
}

namespace intel_mlperf {
int init() {
  return 0;
}
}
