#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <omp.h>

#include "amx_linear.hpp"
#include "i_linear_tpp.hpp"
#include "amx_config.hpp"

namespace intel_mlperf {

at::Tensor amx_linear(
  const at::Tensor& input,
  const at::Tensor& weight,
  const at::Tensor& bias,
  const at::Scalar& scale,
  const bool post_op,
  const at::Scalar& o_scale
) {
  // input shape: [bs, sl, hidden_size]
  auto ishape = input.sizes();
  auto bs = ishape[0];
  auto sl = ishape[1];
  auto hidden_size = ishape[2];

  // weight shape: [col_step, 4, col_tile, 16, 64]
  auto wshape = weight.sizes();
  auto col_step = wshape[0];
  auto col_tile = wshape[2];

  // output shape: [bs, sl, col_step * 64]
  auto output = at::empty({bs, sl, col_step * 64},
                          at::TensorOptions().dtype<int8_t>().memory_format(
                          c10::MemoryFormat::Contiguous));

  auto amx_flag = amx_init();
  if (!amx_flag) {
    return output;
  }

  auto input_ = reinterpret_cast<int8_t (*)[hidden_size]>(input.data_ptr());
  auto weight_ = reinterpret_cast<int8_t (*)[4][col_tile][16][64]>(weight.data_ptr());
  auto output_ = reinterpret_cast<int8_t (*)[col_step][64]>(output.data_ptr());
  auto bias_ = reinterpret_cast<float (*)>(bias.data_ptr());
  auto scale_ = scale.toFloat();
  float o_scale_ = post_op ? o_scale.toFloat() : 1.0;

  auto total_sl = bs * sl;
  size_t row_tile = (total_sl + 15) / 16;
  size_t roll_back = row_tile * 16 - total_sl;

  int col_idx = col_tile == 16 ? 0 : 1;
  size_t os_ = col_step * 64;
  auto block_computer = i_linear(sl, hidden_size, os_, true, post_op);
  
  // 4 loop
  // or only omp parallel 
  # pragma omp parallel 
  {
    auto total_core_num = omp_get_num_threads();
    auto core_id = omp_get_thread_num();
    // printf("------------ core_id : %d / %d\n", core_id, total_core_num);
    size_t start_ = total_sl * core_id / total_core_num;
    size_t chunk_sl_ = (total_sl * core_id + total_sl) / total_core_num - total_sl * core_id / total_core_num;
    block_computer.tile_dot_product_16x256(output_[start_], input_[start_], weight_, bias_, scale_, o_scale_, chunk_sl_, col_step);
  }

  return output;
}

}
