#include "preemphasis.hpp"
#include "tpps/preemphasis_tpp.hpp"
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

namespace intel_mlperf {
at::Tensor preemphasis(const at::Tensor &input,
                       const c10::optional<at::Scalar> &coeff) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  auto output =
      at::empty(in_sz, at::TensorOptions().dtype<float>().memory_format(
                           c10::MemoryFormat::Contiguous));
  auto batch = in_sz[0];
  int64_t seq_len = in_sz[1];

  auto *src = static_cast<float *>(input.data_ptr());
  auto *dst = static_cast<float *>(output.data_ptr());
  auto coeff_ = coeff.value_or(0.97f).toFloat();

  if (coeff_ == 0.0f) {
    output = input;
  } else {
#pragma omp parallel for
    for (auto i = 0; i < batch; i++) {
      preemphasis_tpp::ref(&dst[i * seq_len], &src[i * seq_len], seq_len,
                           coeff_);
    }
  }

  return output;
}

} // namespace intel_mlperf
