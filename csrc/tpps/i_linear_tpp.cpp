#include <chrono>

#include "i_linear_tpp.hpp"
#include "amx_config.hpp"

using Time = std::chrono::high_resolution_clock;

namespace intel_mlperf {

template <int row_tile, int col_tile>
void i_linear::compute_block(void* C, size_t ldc, void* A, void* B, float* bias, float scale, bool post_op, float o_scale) {
  Tilecfg().set_config();
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

void i_linear::tile_dot_product_16x256(void *C, void *A, void *B, float *bias, float scale, float o_scale, 
                                       const size_t sl, const size_t col_step, size_t cur_id, size_t total_chunks) {
  int col_idx = cols_in_tile_ == 16 ? 0 : 1;
  auto C_ = reinterpret_cast<int8_t (*)[col_step][64]>(C);
  auto A_ = reinterpret_cast<int8_t (*)[ic_]>(A);
  auto B_ = reinterpret_cast<int8_t (*)[4][cols_in_tile_][16][64]>(B);
  auto bias_ = reinterpret_cast<float (*)[64]>(bias);

  compute_block_t computer_2 = compute_block_tbl_[2][col_idx];
  compute_block_t computer_1 = compute_block_tbl_[1][col_idx];

  size_t row_tile = (sl + 15) / 16;
  size_t roll_back = row_tile * 16 - sl;

  bool odd_tile = row_tile % 2;
  size_t row_step = row_tile / 2 + odd_tile;

  size_t chunk_step = (col_step + total_chunks - 1) / total_chunks;

  if (odd_tile) {
    for (size_t i = 0; i < chunk_step; i++) {
      for (size_t k = 0; k < total_chunks; k++) {
        for (size_t j = 0; j < row_step; j++) {
          size_t rollback_ = (j == row_step - 1) * roll_back;
          auto computer_   = (j == row_step - 1) ? this->compute_block_tbl_[1][col_idx] : this->compute_block_tbl_[2][col_idx];
          auto row_pos = j * 32 - rollback_;
          size_t col_pos = (cur_id + k) % total_chunks + i * total_chunks;
          if (col_pos >= col_step) {
            continue;
          }
          // if (cur_id == 1) {
          //   printf("%d length sequence : %d / %d, innerfunction --> [%d, %d]\n", sl, cur_id, total_chunks, row_pos, col_pos);
          // }
          (this->*computer_)(C_[row_pos][col_pos], oc_, A_[row_pos], B_[col_pos], bias_[col_pos], scale, post_op_, o_scale);
        }
      }
    }
  } else {
    for (size_t i = 0; i < chunk_step; i++) {
      for (size_t k = 0; k < total_chunks; k++) {
        for (size_t j = 0; j < row_step; j++) {
          size_t rollback_ = (j == row_step - 1) * roll_back;
          auto row_pos = j * 32 - rollback_;
          size_t col_pos = (cur_id + k) % total_chunks + i * total_chunks;
          if (col_pos >= col_step) {
            continue;
          }
          // if (cur_id == 1) {
          //   printf("%d length sequence : %d / %d, innerfunction --> [%d, %d]\n", sl, cur_id, total_chunks, row_pos, col_pos);
          // }
          (this->*computer_2)(C_[row_pos][col_pos], oc_, A_[row_pos], B_[col_pos], bias_[col_pos], scale, post_op_, o_scale);
        }
      }
    }
  }
}

void i_linear::tile_linear(const int row_tile, size_t roll_back, const int col_tile, 
                           void *C, void *A, void *B, float *bias, float scale, float o_scale) {
  int col_idx = col_tile == 16 ? 0 : 1;
  int col_step = oc_ / 64;
  auto C_ = reinterpret_cast<int8_t (*)[col_step][64]>(C);
  auto A_ = reinterpret_cast<int8_t (*)[ic_]>(A);
  auto B_ = reinterpret_cast<int8_t (*)[4][col_tile][16][64]>(B);
  auto bias_ = reinterpret_cast<float (*)[64]>(bias);

  compute_block_t computer_2 = compute_block_tbl_[2][col_idx];
  compute_block_t computer_1 = compute_block_tbl_[1][col_idx];

# pragma unroll
  for (int i = 0; i < col_step; i++) {
    int j = 0;
    int cur_pos = 0;
#   pragma unroll
    for (; j < row_tile - 1; j += 2) {
      cur_pos = (j == row_tile - 2) ? j * 16 - roll_back : j * 16;
      (this->*computer_2)(C_[cur_pos][i], oc_, A_[cur_pos], B_[i], bias_[i], scale, post_op_, o_scale);
    }
    if (row_tile % 2 == 1) {
      cur_pos = j * 16 - roll_back;
      (this->*computer_1)(C_[cur_pos][i], oc_, A_[cur_pos], B_[i], bias_[i], scale, post_op_, o_scale);
    }
  }
}

void i_linear::ref(void* output, void* input, void* weight, float* bias, float scale, float o_scale) {
  // suppose input is padding to 16x and have plain formate, no padding maybe added later

  Tilecfg().set_config();
  const size_t row_block = 16;
  size_t row_tile = (sl_ + 15) / row_block;
  
  size_t roll_back = row_tile * 16 - sl_;

  // col_tile = 16 or 64
  // tile_linear(row_tile, roll_back, cols_in_tile_, output, input, weight, bias, scale, o_scale);
}

}
