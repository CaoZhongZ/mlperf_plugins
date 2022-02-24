#include "i_linear_tpp.hpp"

namespace intel_mlperf {

void block_gemm224::ref(void* output, void* input, void* weight, void* bias, float scale) {
  // suppose input is padding to 16x and have tile formate
  size_t row_block = 16;
  size_t block_num = (dim0 + 15) / 16;
  size_t remainder = block_num % 2;
  size_t row_step = block_num / 2 - remainder;

  size_t col_tile = dim1 / 64;
  size_t col_step = dim2 * 16 / dim1;
  auto input_ = reinterpret_cast<int8_t (*)[col_tile][16][64]>(input);
  auto weight_ = reinterpret_cast<int8_t (*)[dim1 / 4][256]>(weight);
  auto output_ = reinterpret_cast<int8_t (*)[col_step][16][64]>(output);
  auto bias_ = reinterpret_cast<float (*)[64]>(bias);
  size_t i = 0;
  for (i; i < row_step; i++) {
    for (size_t j = 0; j < col_step; j++) {
      _tile_dot_product_16x256<2, 16>::compute(output_[i * 2][j], input_[i * 2], weight_[j], bias_[j], scale);
    }
  }
  if (remainder == 1) {
    for (size_t j = 0; j < col_step; j++) {
      _tile_dot_product_16x256<1, 16>::compute(output_[i * 2][j], input_[i * 2], weight_[j], bias_[j], scale);
    }
  }
}

}