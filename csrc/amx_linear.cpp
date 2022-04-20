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
  // Tilecfg().set_config();

  auto input_ = reinterpret_cast<int8_t (*)[hidden_size]>(input.data_ptr());
  auto weight_ = reinterpret_cast<int8_t (*)[4][col_tile][16][64]>(weight.data_ptr());
  auto output_ = reinterpret_cast<int8_t (*)[col_step][64]>(output.data_ptr());
  auto bias_ = reinterpret_cast<float (*)[64]>(bias.data_ptr());
  auto scale_ = scale.toFloat();
  float o_scale_ = post_op ? o_scale.toFloat() : 1.0;

  auto total_sl = bs * sl;
  size_t row_tile = (total_sl + 15) / 16;
  size_t roll_back = row_tile * 16 - total_sl;

  int col_idx = col_tile == 16 ? 0 : 1;
  auto block_computer = i_linear(sl, hidden_size, col_step * 64, true, post_op);
  auto computer_2 = block_computer.compute_block_tbl_[2][col_idx];
  auto computer_1 = block_computer.compute_block_tbl_[1][col_idx];

# pragma omp parallel for collapse(2)
  for (size_t i = 0; i < col_step; i++) {
    for (size_t j = 0; j < row_tile / 2; j++) {
      if (row_tile % 2 == 0) {
        int cur_pos = (j == row_tile / 2 - 1) ? j * 32 - roll_back : j * 32;
        (block_computer.*computer_2)(output_[cur_pos][i], col_step * 64, input_[cur_pos], weight_[i], bias_[i], scale_, post_op, o_scale.toFloat());
      } else {
        int cur_pos = j * 32;
        (block_computer.*computer_2)(output_[cur_pos][i], col_step * 64, input_[cur_pos], weight_[i], bias_[i], scale_, post_op, o_scale.toFloat());
        if (j == row_tile / 2 - 1) {
          cur_pos += 32 - roll_back;
          (block_computer.*computer_1)(output_[cur_pos][i], col_step * 64, input_[cur_pos], weight_[i], bias_[i], scale_, post_op, o_scale.toFloat());
        }
      }
    }
  }

  return output;
}

}
