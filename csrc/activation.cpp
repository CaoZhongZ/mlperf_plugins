#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>

#include "i_gelu_tpp.hpp"

namespace intel_mlperf {
at::Tensor i_gelu (const at::Tensor& input, double M, double oscale) {
  auto sizes = input.sizes();

  auto batch = sizes[0] * sizes[1];
  auto line  = sizes[2];

  auto output = at::empty(sizes,
      at::TensorOptions().dtype<int8_t>()
      .memory_format(c10::MemoryFormat::Contiguous));

  auto *in = input.data_ptr();
  auto *out = output.data_ptr();

# pragma omp parallel for
  for (auto b = 0; b < batch; ++b) {
    // Move out causing Apple Clang crash
    auto pin = reinterpret_cast<int32_t (*)[line]>(in);
    auto pout = reinterpret_cast<int8_t (*)[line]>(out);

    i_gelu_tpp<16>::ref(pout[b], pin[b], M, oscale, line);
  }

  return output;
}
}
