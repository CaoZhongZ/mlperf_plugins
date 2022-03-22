#include <chrono>

#include "i_linear_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

namespace intel_mlperf {

template <int row_tile, int col_tile>
void i_linear::compute_block(void* C, size_t ldc, void* A, void* B, float* bias, float scale) {
  _tile_dot_product_16x256<row_tile, col_tile, io_policy<col_tile, i_format::plain>>::compute(C, ldc, A, B, bias, scale);  
}

const i_linear::compute_block_t i_linear::compute_block_tbl_ [12][2] = {
  {
    nullptr, nullptr
  }, {
    nullptr, nullptr
  }, {
    &i_linear::compute_block<2, 16>, &i_linear::compute_block<2, 64>
  }, {
    &i_linear::compute_block<3, 16>, &i_linear::compute_block<3, 64>
  }, {
    &i_linear::compute_block<4, 16>, &i_linear::compute_block<4, 64>
  }, {
    &i_linear::compute_block<5, 16>, &i_linear::compute_block<5, 64>
  }, {
    &i_linear::compute_block<6, 16>, &i_linear::compute_block<6, 64>
  }, {
    &i_linear::compute_block<7, 16>, &i_linear::compute_block<7, 64>
  }, {
    &i_linear::compute_block<8, 16>, &i_linear::compute_block<8, 64>
  }, {
    &i_linear::compute_block<9, 16>, &i_linear::compute_block<9, 64>
  }, {
    &i_linear::compute_block<10, 16>, &i_linear::compute_block<10, 64>
  }, {
    &i_linear::compute_block<11, 16>, &i_linear::compute_block<11, 64>
  }
};

void i_linear::tile_dot_product_16x256(const int row_tile, const int col_tile, 
                                       void *C, size_t ldc, void *A, void *B, void *bias, float scale) {
  int col_idx = col_tile == 16 ? 0 : 1;

  auto B_ = reinterpret_cast<int8_t (*)[cols_in_tile_ * 16][256]>(B);
  auto bias_ = reinterpret_cast<float (*)[64]>(bias);
  auto C_ = reinterpret_cast<int8_t (*)[16][total_work_][64]>(C);
  auto A_ = reinterpret_cast<int8_t (*)[16][ic_]>(A);

  switch (row_tile) {
  case (3) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[3][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (4) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[4][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (5) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[5][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (6) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[6][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (7) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[7][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (8) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[8][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (9) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[9][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (10) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[10][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (11) :
    // TODO: add pragma parallel
    auto computer_ = compute_block_tbl_[11][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
    }
    break;
  case (12) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[10][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[10][0][i], ldc, A_[10], B_[i], bias_[i], scale);
    }
    break;
  case (13) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[11][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[11][0][i], ldc, A_[11], B_[i], bias_[i], scale);
    }
    break;
  case (14) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[11][col_idx];
    auto computer_2 = compute_block_tbl_[3][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[11][0][i], ldc, A_[11], B_[i], bias_[i], scale);
    }
    break;
  case (15) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[11][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[11][0][i], ldc, A_[11], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[13][0][i], ldc, A_[13], B_[i], bias_[i], scale);
    }
    break;
  case (16) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[10][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[10][0][i], ldc, A_[10], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[12][0][i], ldc, A_[12], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[14][0][i], ldc, A_[14], B_[i], bias_[i], scale);
    }
    break;
  case (17) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[11][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[11][0][i], ldc, A_[11], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[13][0][i], ldc, A_[13], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[15][0][i], ldc, A_[15], B_[i], bias_[i], scale);
    }
    break;
  case (18) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[10][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[10][0][i], ldc, A_[10], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[12][0][i], ldc, A_[12], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[14][0][i], ldc, A_[14], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[16][0][i], ldc, A_[16], B_[i], bias_[i], scale);
    }
    break;
  case (19) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[11][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[11][0][i], ldc, A_[11], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[13][0][i], ldc, A_[13], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[15][0][i], ldc, A_[15], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[17][0][i], ldc, A_[17], B_[i], bias_[i], scale);
    }
    break;
  case (20) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[10][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[10][0][i], ldc, A_[10], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[12][0][i], ldc, A_[12], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[14][0][i], ldc, A_[14], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[16][0][i], ldc, A_[16], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[18][0][i], ldc, A_[18], B_[i], bias_[i], scale);
    }
    break;
  case (21) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[11][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[11][0][i], ldc, A_[11], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[13][0][i], ldc, A_[13], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[15][0][i], ldc, A_[15], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[17][0][i], ldc, A_[17], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[19][0][i], ldc, A_[19], B_[i], bias_[i], scale);
    }
    break;
  case (22) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[10][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[10][0][i], ldc, A_[10], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[12][0][i], ldc, A_[12], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[14][0][i], ldc, A_[14], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[16][0][i], ldc, A_[16], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[18][0][i], ldc, A_[18], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[20][0][i], ldc, A_[20], B_[i], bias_[i], scale);
    }
    break;
  case (23) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[11][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[11][0][i], ldc, A_[11], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[13][0][i], ldc, A_[13], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[15][0][i], ldc, A_[15], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[17][0][i], ldc, A_[17], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[19][0][i], ldc, A_[19], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[21][0][i], ldc, A_[21], B_[i], bias_[i], scale);
    }
    break;
  case (24) :
    // TODO: add pragma parallel
    auto computer_1 = compute_block_tbl_[10][col_idx];
    auto computer_2 = compute_block_tbl_[2][col_idx];
    for (int i = 0; i < total_work_; i++) {
      (this->*computer_1)(C_[0][0][i], ldc, A_, B_[i], bias_[i], scale);
      (this->*computer_2)(C_[10][0][i], ldc, A_[10], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[12][0][i], ldc, A_[12], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[14][0][i], ldc, A_[14], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[16][0][i], ldc, A_[16], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[18][0][i], ldc, A_[18], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[20][0][i], ldc, A_[20], B_[i], bias_[i], scale);
      (this->*computer_2)(C_[22][0][i], ldc, A_[22], B_[i], bias_[i], scale);
    }
    break;
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