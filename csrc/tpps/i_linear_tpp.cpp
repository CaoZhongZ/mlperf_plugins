#include <chrono>

#include "i_linear_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

namespace intel_mlperf {

template <int row_tile, int col_tile>
void i_linear::compute_block(void* C, size_t ldc, void* A, void* B, float* bias, float scale) {
  _tile_dot_product_16x256<row_tile, col_tile, io_policy<col_tile, i_format::plain>>::compute(C, ldc, A, B, bias, scale);  
}

const i_linear::compute_block_t i_linear::compute_block_tbl_ [12][2] = {
  { nullptr, nullptr }, 
  { &i_linear::compute_block<1, 16>, &i_linear::compute_block<1, 64> }, 
  { &i_linear::compute_block<2, 16>, &i_linear::compute_block<2, 64> }, 
  { &i_linear::compute_block<3, 16>, &i_linear::compute_block<3, 64> }, 
  { &i_linear::compute_block<4, 16>, &i_linear::compute_block<4, 64> }, 
  { &i_linear::compute_block<5, 16>, &i_linear::compute_block<5, 64> }, 
  { &i_linear::compute_block<6, 16>, &i_linear::compute_block<6, 64> }, 
  { &i_linear::compute_block<7, 16>, &i_linear::compute_block<7, 64> }, 
  { &i_linear::compute_block<8, 16>, &i_linear::compute_block<8, 64> }, 
  { &i_linear::compute_block<9, 16>, &i_linear::compute_block<9, 64> }, 
  { &i_linear::compute_block<10, 16>, &i_linear::compute_block<10, 64> }, 
  { &i_linear::compute_block<11, 16>, &i_linear::compute_block<11, 64> }
};

void i_linear::tile_dot_product_16x256(const int row_tile, const int col_tile, 
                                       void *C, size_t ldc, void *A, void *B, void *bias, float scale) {
  int col_idx = col_tile == 16 ? 0 : 1;

  auto B_ = reinterpret_cast<int8_t (*)[cols_in_tile_ * 16][256]>(B);
  auto bias_ = reinterpret_cast<float (*)[64]>(bias);
  auto C_ = reinterpret_cast<int8_t (*)[16][total_work_][64]>(C);
  auto A_ = reinterpret_cast<int8_t (*)[16][ic_]>(A);

  compute_block_t computer_2 = compute_block_tbl_[2][col_idx];
  compute_block_t computer_1 = compute_block_tbl_[1][col_idx];

# pragma unroll
  for (int i = 0; i < total_work_; i++) {
    int j = 0;
#   pragma unroll
    for (; j < row_tile / 2; j++) {
      (this->*computer_2)(C_[j * 2][0][i], ldc, A_[j * 2], B_[i], bias_[i], scale);
    }
    if (row_tile % 2 == 1) {
      (this->*computer_1)(C_[j * 2][0][i], ldc, A_[j * 2], B_[i], bias_[i], scale);
    }
  }
}

void i_linear::ref(void* output, void* input, void* weight, void* bias, float scale) {
  // suppose input is padding to 16x and have plain formate, no padding maybe added later
  size_t row_block = 16;
  size_t row_tile = (sl_ + 15) / 16;
  
  // TODO: add rollback

  // col_tile = 16 or 64
  tile_dot_product_16x256(row_tile, cols_in_tile_, output, oc_, input, weight, bias, scale);
}

}