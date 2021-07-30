#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <immintrin.h>

#include "i_softmax_tpp.hpp"

namespace intel_mlperf {
at::Tensor i_softmax(
    const at::Tensor& input,
    const at::Tensor& att_mask,
    const at::Scalar& M,
    const at::Scalar& oscale) {

  auto in_sz = input.sizes();
  auto batch = std::accumulate(
      in_sz.begin(), in_sz.end() - 2, 1, std::multiplies<int64_t>());

  i_softmax_tpp<16> compute(batch, *(in_sz.end() -2), *(in_sz.end()-1));

  // We only have ref yet.
  auto output = at::empty(in_sz,
      at::TensorOptions().dtype<int8_t>()
      .memory_format(c10::MemoryFormat::Contiguous));

  compute.ref(
      output.data_ptr(), input.data_ptr(),
      att_mask.data_ptr(), M.toFloat(), oscale.toFloat());

  return output;
}
}
