#include "normalization.hpp"
#include "i_layernorm_tpp.hpp"
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

namespace intel_mlperf {
at::Tensor i_layernorm (
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double oscale, c10::optional<double> eps) {
  auto in_sz = input.sizes();
  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2];
  auto output = at::empty(in_sz,
      at::TensorOptions().dtype<int8_t>()
      .memory_format(c10::MemoryFormat::Contiguous));

  auto* in = input.data_ptr();
  auto* w = weight.data_ptr();
  auto* b = bias.data_ptr();
  auto* out = output.data_ptr();

  auto data_type = input.scalar_type();

  if (data_type == c10::ScalarType::Char) {
#   pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto* pin = reinterpret_cast<int8_t (*)[reduce_l]>(in);
      auto* pw = reinterpret_cast<float *>(w);
      auto* pb = reinterpret_cast<float *>(b);
      auto* pout = reinterpret_cast<int8_t (*)[reduce_l]>(out);

      i_layernorm_tpp<16>::ref(
          pout[i], pin[i], pw, pb, oscale, reduce_l, eps.value_or(1e-5));
    }
  } else if (data_type == c10::ScalarType::Float) {
#   pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto* pin = reinterpret_cast<float (*)[reduce_l]>(in);
      auto* pw = reinterpret_cast<float *>(w);
      auto* pb = reinterpret_cast<float *>(b);
      auto* pout = reinterpret_cast<int8_t (*)[reduce_l]>(out);

      i_layernorm_tpp<16>::ref(
          pout[i], pin[i], pw, pb, oscale, reduce_l, eps.value_or(1e-5));
    }
  } // throw here

  return output;
}

at::Tensor i_residual_layernorm (
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double scale_1, double scale_2,
    double oscale, c10::optional<double> eps) {
  auto in_sz = input1.sizes();

  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2];

  auto output = at::empty(in_sz,
      at::TensorOptions().dtype<int8_t>()
      .memory_format(c10::MemoryFormat::Contiguous));

  auto* src1 = input1.data_ptr();
  auto* src2 = input2.data_ptr();
  auto* w = weight.data_ptr();
  auto* b = bias.data_ptr();
  auto* out = output.data_ptr();

# pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    auto* psrc1 = reinterpret_cast<int8_t (*)[reduce_l]>(src1);
    auto* psrc2 = reinterpret_cast<int8_t (*)[reduce_l]>(src2);
    auto* pw = reinterpret_cast<float *>(w);
    auto* pb = reinterpret_cast<float *>(b);
    auto* pout = reinterpret_cast<int8_t (*)[reduce_l]>(out);

    i_residual_layernorm_tpp<16>::ref(pout[i], psrc1[i], psrc2[i], pw, pb,
        scale_1, scale_2, oscale, reduce_l, eps.value_or(1e-5));
  }

  return output;
}
}
