#include "power_spectrum.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "tpps/power_spectrum_tpp.hpp"

namespace intel_mlperf {
at::Tensor power_spectrum(const at::Tensor &input) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  auto batch = in_sz[0];
  auto seq_len = in_sz[1];
  auto output = at::empty(
      {batch, seq_len},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto src = input.accessor<float, 3>();
  auto dst = output.accessor<float, 2>();

#pragma omp parallel for
  for (auto i = 0; i < batch; i++) {
    power_spectrum_tpp::ref(&dst[i][0], &src[i][0][0], seq_len);
  }

  return output;
}

}  // namespace intel_mlperf
