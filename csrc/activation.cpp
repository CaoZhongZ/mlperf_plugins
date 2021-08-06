#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>

#include "i_gelu_tpp.hpp"

namespace intel_mlperf {
at::Tensor i_gelu (
    const at::Tensor& input,
    const at::Scalar& M,
    const at::Scalar& oscale) {
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
    // Move out will cause Apple Clang crash
    auto pin = reinterpret_cast<int32_t (*)[line]>(in);
    auto pout = reinterpret_cast<int8_t (*)[line]>(out);

    i_gelu_tpp<16>::ref(pout[b], pin[b], M.toFloat(), oscale.toFloat(), line);
  }

  return output;
}

at::Tensor i_identity(
    const at::Tensor& input,
    const c10::optional<at::Scalar>& M,
    const at::Scalar& oscale) {
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
    // Move out will cause Apple Clang crash
    auto pin = reinterpret_cast<float (*)[line]>(in);
    auto pout = reinterpret_cast<int8_t (*)[line]>(out);

    i_identity_tpp<16>::ref(pout[b], pin[b], oscale.toFloat(), line);
  }

  return output;

}

at::Tensor i_identity_(
    at::Tensor& self,
    const c10::optional<at::Scalar>& M,
    const at::Scalar& oscale) {
  auto sizes = self.sizes();

  auto batch = sizes[0] * sizes[1];
  auto line  = sizes[2];

  auto *in = self.data_ptr();
  auto *out = in;

# pragma omp parallel for
  for (auto b = 0; b < batch; ++b) {
    // Move out will cause Apple Clang crash
    auto pin = reinterpret_cast<float (*)[line]>(in);
    auto pout = reinterpret_cast<int8_t (*)[line]>(out);

    i_identity_tpp<16>::ref(pout[b], pin[b], oscale.toFloat(), line);
  }

  return self;
}

}
