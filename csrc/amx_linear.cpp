#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>

#include "amx_linear.hpp"
#include "i_linear_tpp.hpp"
#include "amx_config.hpp"

namespace intel_mlperf {

at::Tensor amx_linear(
  const at::Tensor& input,
  const at::Tensor& weight,
  const at::Tensor& bias,
  const at::Scalar& scale
) {
  auto ishape = input.sizes();
  auto bs = ishape[0];
  auto i_row_bn = ishape[1];
  auto i_col_bn = ishape[2];

  auto wshape = weight.sizes();
  auto w_col_bn = wshape[0];
  auto w_row_bn = wshape[2];

  auto output = at::empty({bs, i_row_bn, w_col_bn, 16, 64},
                          at::TensorOptions().dtype<int8_t>().memory_format(
                          c10::MemoryFormat::Contiguous));

  auto amx_flag = amx_init();
  if (!amx_flag) {
    return output;
  }
  Tilecfg().set_config();

  auto input_ = reinterpret_cast<int8_t (*)[i_row_bn][i_col_bn][16][64]>(input.data_ptr());
  auto weight_ = reinterpret_cast<int8_t (*)[4][w_row_bn][16][64]>(weight.data_ptr());
  auto output_ = reinterpret_cast<int8_t (*)[i_row_bn][w_col_bn][16][64]>(output.data_ptr());
  auto bias_ = bias.data_ptr();
  auto scale_ = scale.toFloat();

  size_t dim0 = i_row_bn * 16;
  size_t dim1 = i_col_bn * 64;
  size_t dim2 = w_col_bn * 64;
  block_gemm<4> gemm_compute(dim0, dim1, dim2);

  for (size_t i = 0; i < bs; i++) {
    gemm_compute.ref(output_[i], input_[i], weight_, bias_, scale_);
  } 
  return output;
}

}