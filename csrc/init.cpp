#include <torch/library.h>
#include <plugins.hpp>
#include "linear.hpp"

TORCH_LIBRARY(intel_mlperf, m) {
  m.def(
    "linear(Tensor input, Tensor weight, Tensor ? bias, float ? scale, int ? zero) -> Tensor",
    intel_mlperf::linear);
  m.def("linear_add_(Tensor(a!) self, Tensor input, Tensor weight, Tensor ? bias, float ? scale, int ? zero) -> Tensor (a!)",
    intel_mlperf::linear_add_);

  m.def(
      "prepack_linear_weight(Tensor weight) -> Tensor",
      intel_mlperf::prepack_linear_weight);

  m.def("baddbmm_out_", intel_mlperf::baddbmm_out_);
  m.def("matmul_out_", intel_mlperf::matmul_out_);
  m.def("reorder_test(Tensor input) -> Tensor", intel_mlperf::reorder_test);
}

namespace intel_mlperf {
int init() {
  return 0;
}
}
