#include "helper.hpp"
#include "i_softmax_tpp.hpp"
#include "transpose.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <random>
#include <cmath>

#include "i_softmax_tpp.hpp"
#include "i_linear_tpp.hpp"
#include "amx_config.hpp"

using Time = std::chrono::high_resolution_clock;

void set_data_act(void *a, size_t n_tile, size_t hidden_length = 1024)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(seed);
  std::normal_distribution<double> dis(0, 1);
  auto a_ = reinterpret_cast<int8_t (*)>(a);
  size_t elenum = n_tile * 16 * hidden_length;
  for (int i = 0; i < elenum; i++) {
    a_[i] = static_cast<int8_t>(dis(gen) * 0xf);
    // a_[i] = 0;
  }
}

void set_data_wei(void *w, void* b, size_t col_tile = 16, size_t col_step = 1) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(seed);
  std::normal_distribution<double> dis(0, 1);
  auto w_ = reinterpret_cast<int8_t (*)>(w);
  size_t elenum = col_tile * 16 * 256 * col_step;
  for (int i = 0; i < elenum; i++) {
    w_[i] = static_cast<int8_t>(dis(gen) * 0xf);
    // w_[i] = 0;
  }
  auto b_ = reinterpret_cast<float (*)>(b);
  for (int i = 0; i < 64 * col_step; i++) {
    b_[i] = static_cast<float>(dis(gen) * 0xf);
    // b_[i] = 0;
  }
}

static constexpr size_t cols = 384;
static constexpr size_t rows = 16;
static constexpr int qmax = 127.0;
static constexpr int qmin = -127.0;

void naive_linear(void* a, const size_t lda, void* b, const size_t ldb, void* c, const size_t ldc, void* bias, float scale, int row_tile) {
  // this function is to test _tile_gemm output 64 of ldc
  auto a_ = reinterpret_cast<int (*)[lda]>(a);
  auto b_ = reinterpret_cast<int (*)[ldb]>(b);
  auto c_ = reinterpret_cast<int (*)[ldc]>(c);
  auto bias_ = reinterpret_cast<float (*)>(bias);

  for (int i = 0; i < 16 * row_tile; i++) {
    for (int j = 0; j < ldc; j++) {
      c_[i][j] = 0;
    }
  }

  for (int i = 0; i < 16 * row_tile; i++) {
    for (int j = 0; j < ldc; j++) {
      for (int k = 0; k < lda; k++) {
        c_[i][j] += a_[i][k] * b_[k][j];
        // std::cout << i << " " << k << " " << j << " : " << a_[i][k] << " , " << b_[k][j] << " : " << c_[i][j] << std::endl;
      }
      // getchar();
    }
  }

  for (int i = 0; i < 16 * row_tile; i++) {
    for (int j = 0; j < ldc; j++) {
      float tem = static_cast<float>(c_[i][j]);
      tem += bias_[j];
      tem *= scale;
      c_[i][j] = static_cast<int>(round(tem));
      c_[i][j] = c_[i][j] < qmax ? c_[i][j] : qmax;
      c_[i][j] = c_[i][j] > qmin ? c_[i][j] : qmin;
    }
  }
}

void send_data2naive(void* a, void* b, void* na, void* nb, int row_tile, bool is_block, size_t lda = 1024, size_t ldb = 64) {
  if (is_block) {
    auto a_ = reinterpret_cast<int8_t (*)[16][16][64]>(a);
    auto na_ = reinterpret_cast<int (*)[lda]>(na);
    for (int i = 0; i < row_tile; i++) {
      for (int j = 0; j < 16; j++) {
        for (int k = 0; k < 16; k++) {
          for (int m = 0; m < 64; m++) {
            na_[i * 16 + k][j * 64 + m] = static_cast<int>(a_[i][j][k][m]);
          }
        }
      }
    }
  } else {
    auto a_ = reinterpret_cast<int8_t (*)[lda]>(a);
    auto na_ = reinterpret_cast<int (*)[lda]>(na);
    for (int i = 0; i < row_tile * 16; i++) {
      for (int j = 0; j < lda; j++) {
        na_[i][j] = static_cast<int>(a_[i][j]);
      }
    }
  }
  

  auto b_ = reinterpret_cast<int8_t (*)[16][16][64]>(b);
  auto nb_ = reinterpret_cast<int (*)[ldb]>(nb);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k++) {
        for (int m = 0; m < 64; m++) {
          auto row_offset = m % 4;
          auto col_offset = m / 4;
          nb_[j * 64 + k * 4 + row_offset][i * 16 + col_offset] = static_cast<int>(b_[i][j][k][m]);
        }
      }
    }
  }
}

void send_input(void* input, void* ninput, const size_t dim0, const size_t dim1) {
  auto input_ = reinterpret_cast<int8_t (*)[dim1]>(input);
  auto ninput_ = reinterpret_cast<int (*)[dim1]>(ninput);

  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      ninput_[i][j] = static_cast<int>(input_[i][j]);
    }
  }
}

void send_weight(void* weight, void* nweight, const size_t dim1, const size_t dim2) {
  size_t col_step = dim2 / 64;
  size_t col_tile = dim1 / 64;

  auto weight_ = reinterpret_cast<int8_t (*)[4][col_tile][16][64]>(weight);
  auto nweight_ = reinterpret_cast<int (*)[dim2]>(nweight);
  for (size_t i = 0; i < col_step; i++) {
    for (size_t j = 0; j < 4; j++) {
      for (size_t m = 0; m < col_tile; m++) {
        for (size_t k = 0; k < 16; k++) {
          for (size_t n = 0; n < 64; n++) {
            auto row_offset = n % 4;
            auto col_offset = n / 4;
            nweight_[m * 64 + k * 4 + row_offset][i * 64 + j * 16 + col_offset] = static_cast<int>(weight_[i][j][m][k][n]);
          }
        }
      }
    }
  }
}

void test_accuracy_linear(int row_tile) {
  alignas(64) int8_t act[row_tile][16][1024];
  alignas(64) int8_t wei[256 * 256];
  alignas(64) int nact[row_tile * 16 * 1024];
  alignas(64) int nwei[1024 * 64];
  
  alignas(64) float bias[64];

  size_t ldc = 1024;
  
  alignas(64) int nout[row_tile * 16 * ldc];
  float scale = 0.0018;
  set_data_act(act, row_tile);
  set_data_wei(wei, bias);

  using tile_io  = intel_mlperf::io_policy<16, intel_mlperf::i_format::tile>;

  // first : do tile block
  send_data2naive(act, wei, nact, nwei, row_tile, true);

  // intel_mlperf::amx_init();
  // intel_mlperf::Tilecfg().set_config();

  naive_linear(nact, 1024, nwei, 64, nout, ldc, bias, scale, row_tile);
  printf("****************** start block test row_tile = %d... *********************\n", row_tile);
  printf("****************** accuracy...*********************\n");
  alignas(64) int8_t out[row_tile][16][64];

  switch (row_tile) {
  case (2):
    intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (3):
    intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (4):
    intel_mlperf::_tile_dot_product_16x256<4, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (5):
    intel_mlperf::_tile_dot_product_16x256<5, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (6):
    intel_mlperf::_tile_dot_product_16x256<6, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (7):
    intel_mlperf::_tile_dot_product_16x256<7, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (8):
    intel_mlperf::_tile_dot_product_16x256<8, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (9):
    intel_mlperf::_tile_dot_product_16x256<9, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (10):
    intel_mlperf::_tile_dot_product_16x256<10, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (11):
    intel_mlperf::_tile_dot_product_16x256<11, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  case (12):
    intel_mlperf::_tile_dot_product_16x256<12, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    break;
  }

  auto out_ = reinterpret_cast<int8_t (*)[16][64]>(out);
  auto nout_ = reinterpret_cast<int (*)[ldc]>(nout);
  for (int i = 0; i < row_tile; i++) {
    intel_mlperf::compare_naive_output(&nout_[i * 16][0], (int8_t*)out_[i], 16, 64, ldc, 64);
  } 

  printf("************************ start performance test... **************************\n");
  int count = 64000;
  // auto wei400m = new int8_t[6400 * 256 * 256];
  // auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(wei400m);
  auto lstart = Time::now();
  for (int i = 0; i < count; i++) {
    switch (row_tile) {
    case (2):
      intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (3):
      intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (4):
      intel_mlperf::_tile_dot_product_16x256<4, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (5):
      intel_mlperf::_tile_dot_product_16x256<5, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (6):
      intel_mlperf::_tile_dot_product_16x256<6, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (7):
      intel_mlperf::_tile_dot_product_16x256<7, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (8):
      intel_mlperf::_tile_dot_product_16x256<8, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (9):
      intel_mlperf::_tile_dot_product_16x256<9, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (10):
      intel_mlperf::_tile_dot_product_16x256<10, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (11):
      intel_mlperf::_tile_dot_product_16x256<11, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    case (12):
      intel_mlperf::_tile_dot_product_16x256<12, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
      break;
    }
  }
  auto lduring =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - lstart)
          .count();
  std::cout << count  << " times block tile linear time : "
            << (float)lduring / 1000 / 1000 << " ms " << std::endl;
  std::cout << "single linear time : " << (float)lduring / count << " ns" << std::endl;

  // getchar();
  printf("****************** start plain test... *********************\n");

  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;

  send_data2naive(act, wei, nact, nwei, row_tile, false);
  naive_linear(nact, 1024, nwei, 64, nout, ldc, bias, scale, row_tile);
  int8_t p_out[row_tile * 16][ldc];
  printf("****************** accuracy...*********************\n");
  switch (row_tile) {
  case (2):
    intel_mlperf::_tile_dot_product_16x256<2, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (3):
    intel_mlperf::_tile_dot_product_16x256<3, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (4):
    intel_mlperf::_tile_dot_product_16x256<4, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (5):
    intel_mlperf::_tile_dot_product_16x256<5, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (6):
    intel_mlperf::_tile_dot_product_16x256<6, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (7):
    intel_mlperf::_tile_dot_product_16x256<7, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (8):
    intel_mlperf::_tile_dot_product_16x256<8, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (9):
    intel_mlperf::_tile_dot_product_16x256<9, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (10):
    intel_mlperf::_tile_dot_product_16x256<10, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (11):
    intel_mlperf::_tile_dot_product_16x256<11, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  case (12):
    intel_mlperf::_tile_dot_product_16x256<12, 16, plain_io>::compute(p_out, ldc, act, wei, bias, scale);
    break;
  }

  auto p_out_ = reinterpret_cast<int8_t (*)[16][ldc]>(p_out);
  nout_ = reinterpret_cast<int (*)[ldc]>(nout);
  for (int i = 0; i < row_tile; i++) {
    intel_mlperf::compare_naive_output(&nout_[i * 16][0], (int8_t*)p_out_[i], 16, 64, ldc, ldc);
  } 

  auto wei400m = new int8_t[6400 * 256 * 256];
  auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(wei400m);
  for (int i = 0; i < 6400; i++) {
    set_data_wei(wei400m_[i], bias);
  }
  count = 6400000;
  printf("************************ start performance test... **************************\n");
  lstart = Time::now();
  for (int i = 0; i < count; i++) {
    switch (row_tile) {
    case (2):
      intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (3):
      intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (4):
      intel_mlperf::_tile_dot_product_16x256<4, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (5):
      intel_mlperf::_tile_dot_product_16x256<5, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (6):
      intel_mlperf::_tile_dot_product_16x256<6, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (7):
      intel_mlperf::_tile_dot_product_16x256<7, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      // intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      // intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(p_out[2], ldc, act[2], wei400m_[i % 6400], bias, scale);
      // intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(p_out[4], ldc, act[4], wei400m_[i % 6400], bias, scale);
      break;
    case (8):
      intel_mlperf::_tile_dot_product_16x256<8, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (9):
      intel_mlperf::_tile_dot_product_16x256<9, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (10):
      intel_mlperf::_tile_dot_product_16x256<10, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (11):
      intel_mlperf::_tile_dot_product_16x256<11, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    case (12):
      intel_mlperf::_tile_dot_product_16x256<12, 16, tile_io>::compute(p_out, ldc, act, wei400m_[i % 6400], bias, scale);
      break;
    }
  }
  lduring =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - lstart)
          .count();
  std::cout << count  << " times plain tile linear time : "
            << (float)lduring / 1000 / 1000 << " ms " << std::endl;
  std::cout << "single linear time : " << (float)lduring / count << " ns" << std::endl;

  delete[] wei400m;
}

void test_block_gemm(const size_t dim0, const size_t dim1, const size_t dim2, bool accuracy = true) {
  auto gemm_ = intel_mlperf::i_linear(dim0, dim1, dim2);

  size_t row_tile = (dim0 + 15) / 16;
  size_t col_step = dim2 / 64;
  alignas(64) int8_t input[row_tile][16][dim1];
  // int8_t weight[dim2 / 64][dim1 / 4][256];
  auto weight400m = new int8_t[400 * dim1 * dim2];
  auto weight400m_ = reinterpret_cast<int8_t (*)[dim1 * dim2]>(weight400m);
  auto weight = weight400m_[0];
  alignas(64) int8_t output[dim0][dim2];
  float bias[col_step][64];
  float scale = 0.0018;
  set_data_act(input, row_tile, dim1);
  for (int i = 0; i < 400; i++) {
    set_data_wei(weight400m_[i], bias, dim1 / 64, col_step);
  }
  

  // intel_mlperf::print_2d_matrix<int>((int*)nweight, dim1, dim2, 1024);
  // getchar();

  // intel_mlperf::print_2d_matrix<int>((int*)noutput, dim0, dim2, dim2);
  // getchar();
  // intel_mlperf::print_2d_matrix<int8_t>((int8_t*)output, dim0, dim2, dim2);
  // getchar();
  if (accuracy) {
    switch (dim1 / 64) {
    case (16):
      gemm_.ref<16>(output, input, weight, bias, scale);
      break;
    case (64):
      gemm_.ref<64>(output, input, weight, bias, scale);
      break;
    }

    auto ninput = new int[dim0 * dim1];
    auto nweight = new int[dim1 * dim2];
    auto noutput = new int[dim0 * dim2];

    send_input(input, ninput, dim0, dim1);
    send_weight(weight, nweight, dim1, dim2);
    naive_linear(ninput, dim1, nweight, dim2, noutput, dim2, bias, scale, row_tile);
    intel_mlperf::compare_naive_output((int*)noutput, (int8_t*)output, dim0, dim2, dim2, dim2);

    delete[] ninput;
    delete[] nweight;
    delete[] noutput;
  } else {
    printf("**************************** start test performance **********************\n");
    auto start = Time::now();
    for (int i = 0; i < 40000; i++) {
      switch (dim1 / 64) {
      case (16):
        gemm_.ref<16>(output, input, weight400m_[i % 400], bias, scale);
        break;
      case (64):
        gemm_.ref<64>(output, input, weight400m_[i % 400], bias, scale);
        break;
      }
    }
    auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
    std::cout << dim0 << " x " << dim1 << " x " << dim2 << " : " << (float)during / 1000 / 1000 / 40000 << " ms " << std::endl;
  }
  delete[] weight400m;
}

int main(int argc, char* argv[]) {
  int row_tile = 2;
  int is_block = 1;
  if (argc == 2) {
    row_tile = std::atoi(argv[1]);
  }
  intel_mlperf::amx_init();
  intel_mlperf::Tilecfg().set_config();

  bool accuracy_mode = false;
  
  test_accuracy_linear(row_tile);
  // for (int i = 7; i <= 24; i++) {
  //   // printf("************************ 1024x1024 test row_tile: %d ********************\n", i);
  //   test_block_gemm(i * 16, 1024, 1024, accuracy_mode);
  // }

  // for (int i = 3; i <= 24; i++) {
  //   // printf("************************ 1024x4096 test row_tile: %d ********************\n", i);
  //   test_block_gemm(i * 16, 1024, 4096, accuracy_mode);
  // }

  // for (int i = 3; i <= 24; i++) {
  //   // printf("************************ 4096x1024 test row_tile: %d ********************\n", i);
  //   test_block_gemm(i * 16, 4096, 1024, accuracy_mode);
  // }
  return 0;
}
