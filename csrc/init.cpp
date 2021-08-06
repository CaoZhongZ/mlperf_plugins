#include <torch/library.h>
#include <plugins.hpp>

#include "linear.hpp"
#include "softmax.hpp"
#include "activation.hpp"
#include "normalization.hpp"

TORCH_LIBRARY(intel_mlperf, m) {
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
      "i_gelu(Tensor input, Scalar M, Scalar oscale) -> Tensor",
      intel_mlperf::i_gelu);
  m.def(
      "i_identity(Tensor input, Scalar ? M, Scalar oscale) -> Tensor",
      intel_mlperf::i_identity);
  m.def(
      "i_identity_(Tensor(a!) input, Scalar ? M, Scalar oscale) -> Tensor(a!)",
      intel_mlperf::i_identity_);
  m.def(
      "i_residual_layernorm(Tensor input1, Tensor input2, Tensor weight, Tensor bias, Scalar scale_1, Scalar scale_2, Scalar oscale, Scalar ? eps) -> Tensor",
      intel_mlperf::i_residual_layernorm);
  m.def(
      "i_residual_layernorm_(Tensor(a!) input1, Tensor input2, Tensor weight, Tensor bias, Scalar scale_1, Scalar scale_2, Scalar oscale, Scalar ? eps) -> Tensor(a!)",
      intel_mlperf::i_residual_layernorm_);
  m.def(
      "i_layernorm(Tensor input, Tensor weight, Tensor bias, Scalar oscale, Scalar ? eps) -> Tensor",
      intel_mlperf::i_layernorm);
}

namespace intel_mlperf {
int init() {
  return 0;
}
}
