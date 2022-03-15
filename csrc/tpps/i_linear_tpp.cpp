#include <chrono>

#include "i_linear_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

namespace intel_mlperf {

template<int col_tile>
void block_gemm::ref(void* output, void* input, void* weight, void* bias, float scale) {
  // suppose input is padding to 16x and have plain formate, no padding maybe added later
  size_t row_block = 16;
  size_t row_tile = (dim0 + 15) / 16;
  // TODO: add rollback

  // col_tile = 16 or 64
  size_t col_step = dim2 / 64;

  auto weight_ = reinterpret_cast<int8_t (*)[col_tile * 16][256]>(weight);
  auto bias_ = reinterpret_cast<float (*)[64]>(bias);
  auto output_ = reinterpret_cast<int8_t (*)[col_step][64]>(output);
  auto input_ = reinterpret_cast<int8_t (*)[dim1]>(input);
  using ip = io_policy<col_tile, i_format::plain>;
  switch (row_tile) {
  case (3) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<3, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (4) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<4, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (5) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<5, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (6) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<6, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (7) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<7, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (8) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<8, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (9) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<9, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (10) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<10, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (11) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
    }
    break;
  case (12) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<10, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[10 * 16][i], dim2, input_[10 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (13) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (14) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<3, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (15) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (16) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<3, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (17) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (18) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<3, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (19) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[17 * 16][i], dim2, input_[17 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (20) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<3, col_tile, ip>::compute(output_[17 * 16][i], dim2, input_[17 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (21) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[17 * 16][i], dim2, input_[17 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[19 * 16][i], dim2, input_[19 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (22) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[17 * 16][i], dim2, input_[17 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<3, col_tile, ip>::compute(output_[19 * 16][i], dim2, input_[19 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (23) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[17 * 16][i], dim2, input_[17 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[19 * 16][i], dim2, input_[19 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[21 * 16][i], dim2, input_[21 * 16], weight_[i], bias_[i], scale);
    }
    break;
  case (24) :
    // TODO: add pragma parallel
    for (int i = 0; i < col_step; i++) {
      _tile_dot_product_16x256<11, col_tile, ip>::compute(output_[0][i], dim2, input, weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[11 * 16][i], dim2, input_[11 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[13 * 16][i], dim2, input_[13 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[15 * 16][i], dim2, input_[15 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[17 * 16][i], dim2, input_[17 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<2, col_tile, ip>::compute(output_[19 * 16][i], dim2, input_[19 * 16], weight_[i], bias_[i], scale);
      _tile_dot_product_16x256<3, col_tile, ip>::compute(output_[21 * 16][i], dim2, input_[21 * 16], weight_[i], bias_[i], scale);
    }
    break;
  }

}

template void block_gemm::ref<16>(void* output, void* input, void* weight, void* bias, float scale);
template void block_gemm::ref<64>(void* output, void* input, void* weight, void* bias, float scale);

}