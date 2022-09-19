#include "frame_splicing.hpp"
#include "tpps/frame_splicing_tpp.hpp"
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <stdexcept>

namespace intel_mlperf {
at::Tensor frame_splicing(const at::Tensor &input, const at::Scalar &factor) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  // N * F * T -> N * (F*factor) * (T//factor)
  auto batch = in_sz[0];
  auto freq = in_sz[1];
  auto time = in_sz[2];
  auto factor_ = factor.toInt();
  auto padded_time = ((time - 1) / factor_) + 1;
  auto output = at::empty({batch, freq * factor_, padded_time},
                          at::TensorOptions().dtype<float>().memory_format(
                              c10::MemoryFormat::Contiguous));

  auto in = input.accessor<float, 3>();
  auto out = output.accessor<float, 3>();

  if (factor_ == 3) {
#pragma omp parallel for collapse(2)
    for (auto i = 0; i < batch; i++) {
      for (int32_t j = 0; j < freq; j++) {
        frame_splicing_tpp<3>::ref(&out[i][j][0], &in[i][0][0], j, freq, time);
      }
    }
  } else {
    throw std::runtime_error("Unsupported frame splicing factor.");
  }

  return output;
}

} // namespace intel_mlperf
