#include <chrono>

#include "i_linear_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

namespace intel_mlperf {

template <int row_tile, int col_tile>
void i_linear::compute_block(void* C, size_t ldc, void* A, void* B, float* bias, float scale, bool post_op, float o_scale) {
  _tile_dot_product_16x256<row_tile, col_tile, io_policy<col_tile, i_format::plain>>::compute(C, ldc, A, B, bias, scale, post_op, o_scale);  
}

const i_linear::compute_block_t i_linear::compute_block_tbl_ [3][2] = {
  { nullptr, nullptr }, 
  { &i_linear::compute_block<1, 16>, &i_linear::compute_block<1, 64> }, 
  { &i_linear::compute_block<2, 16>, &i_linear::compute_block<2, 64> }, 
  // { &i_linear::compute_block<3, 16>, &i_linear::compute_block<3, 64> }, 
  // { &i_linear::compute_block<4, 16>, &i_linear::compute_block<4, 64> }, 
  // { &i_linear::compute_block<5, 16>, &i_linear::compute_block<5, 64> }, 
  // { &i_linear::compute_block<6, 16>, &i_linear::compute_block<6, 64> }, 
  // { &i_linear::compute_block<7, 16>, &i_linear::compute_block<7, 64> }, 
  // { &i_linear::compute_block<8, 16>, &i_linear::compute_block<8, 64> }, 
  // { &i_linear::compute_block<9, 16>, &i_linear::compute_block<9, 64> }, 
  // { &i_linear::compute_block<10, 16>, &i_linear::compute_block<10, 64> }, 
  // { &i_linear::compute_block<11, 16>, &i_linear::compute_block<11, 64> }
};

void i_linear::tile_dot_product_16x256(const int row_tile, size_t roll_back, const int col_tile, 
                                       void *C, size_t ldc, void *A, void *B, float *bias, float scale, float o_scale) {
  int col_idx = col_tile == 16 ? 0 : 1;

  auto C_ = reinterpret_cast<int8_t (*)[ldc]>(C);
  auto A_ = reinterpret_cast<int8_t (*)[ic_]>(A);

  compute_block_t computer_2 = compute_block_tbl_[2][col_idx];
  compute_block_t computer_1 = compute_block_tbl_[1][col_idx];

  int j = 0;
  int cur_pos = 0;
# pragma unroll
  for (; j < row_tile - 1; j += 2) {
    cur_pos = (j == row_tile - 2) ? j * 16 - roll_back : j * 16;
    (this->*computer_2)(C_[cur_pos], ldc, A_[cur_pos], B, bias, scale, post_op_, o_scale);
  }
  if (row_tile % 2 == 1) {
    cur_pos = j * 16 - roll_back;
    (this->*computer_1)(C_[cur_pos], ldc, A_[cur_pos], B, bias, scale, post_op_, o_scale);
  }
}

void i_linear::ref(void* output, void* input, void* weight, float* bias, float scale, float o_scale) {
  // suppose input is padding to 16x and have plain formate, no padding maybe added later
  const size_t row_block = 16;
  size_t row_tile = (sl_ + 15) / row_block;
  
  size_t roll_back = row_tile * 16 - sl_;

  // col_tile = 16 or 64
  tile_dot_product_16x256(row_tile, roll_back, cols_in_tile_, output, oc_, input, weight, bias, scale, o_scale);
}

}