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
#include <math.h>
#include <omp.h>

#include "i_softmax_tpp.hpp"
#include "i_linear_tpp.hpp"
#include "amx_config.hpp"
#include "../amx_init.hpp"

#include "test_gemm.h"

float gelu_func(float x) {
  float rsqrt_2 = 0.70710678;
  auto y = std::erf(x * rsqrt_2) + 1;
  return x * y * 0.5;
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

void performance_test_256x256(int row_tile, void* C, size_t ldc, void* A, void* B, float* bias, float scale, 
                              size_t counter, size_t core_num, bool post_op = false, float o_scale = 1.0) {

  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;
  int loop_num = row_tile / 2;
  int remaining = row_tile % 2;
  size_t single_loop = counter / core_num;

# pragma omp parallel for collapse(1) num_threads (core_num)
  for (size_t i = 0; i < counter; i++) {
#   pragma unroll
    for (int j = 0; j < loop_num; j++) {
      auto core_out = reinterpret_cast<int8_t (*)[row_tile * 16][ldc]>(C);
      auto act = reinterpret_cast<int8_t (*)[row_tile * 16][ldc]>(A);
      auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(B);

      // intel_mlperf::_tile_dot_product_16x256<2, 16, plain_io>::compute(core_out[i / single_loop][j * 2 * 16], ldc, act[i / single_loop][j * 2], wei400m_[i % counter], bias, scale);
      intel_mlperf::_tile_dot_product_16x256<2, 16, plain_io>::compute(core_out[i / single_loop][j * 2 * 16], ldc, act[i / single_loop][j * 2], B, bias, scale, post_op, o_scale);
    }
    // if (remaining) {
    //   intel_mlperf::_tile_dot_product_16x256<1, 16, plain_io>::compute(core_out[i / single_loop][loop_num * 2 * 16], ldc, act[i / single_loop][loop_num * 2], wei400m_[i % counter], bias, scale);
    // }
  }
}

void accuracy_test_256x256(int row_tile, void* C, size_t ldc, void* A, void* B, float* bias, float scale, bool post_op = false, float o_scale = 1.0) {

  using plain_io = intel_mlperf::io_policy<16, intel_mlperf::i_format::plain>;
  int loop_num = row_tile / 2;
  int remaining = row_tile % 2;

  auto C_ = reinterpret_cast<int8_t (*)[ldc]>(C);
  auto act = reinterpret_cast<int8_t (*)[ldc]>(A);
  
  for (int j = 0; j < row_tile - 1; j += 2) {
    intel_mlperf::_tile_dot_product_16x256<2, 16, plain_io>::compute(C_[j * 16], ldc, act[j * 16], B, bias, scale, post_op, o_scale);
  }
  if (remaining) {
    intel_mlperf::_tile_dot_product_16x256<1, 16, plain_io>::compute(C_[loop_num * 2 * 16], ldc, act[loop_num * 2 * 16], B, bias, scale, post_op, o_scale);
  }
}

void test_tile_16x256(int row_tile) {
  alignas(64) int8_t act[row_tile][16][1024];
  alignas(64) int8_t wei[256 * 256];
  alignas(64) int nact[row_tile * 16 * 1024];
  alignas(64) int nwei[1024 * 64];
  
  alignas(64) float bias[64];

  size_t ldc = 1024;
  
  alignas(64) int nout[row_tile * 16 * ldc];
  float scale = 0.0018;

  intel_mlperf::set_data_act(act, row_tile * 16);
  intel_mlperf::set_data_wei(wei, bias);

  using tile_io  = intel_mlperf::io_policy<16, intel_mlperf::i_format::tile>;

  // first : do tile block
  send_data2naive(act, wei, nact, nwei, row_tile, true);

  // intel_mlperf::amx_init();
  // intel_mlperf::Tilecfg().set_config();

  intel_mlperf::naive_linear(nact, 1024, nwei, 64, nout, ldc, bias, scale, row_tile * 16);
  printf("****************** start block test row_tile = %d... *********************\n", row_tile);
  printf("****************** accuracy...*********************\n");
  alignas(64) int8_t out[row_tile][16][64];

  // switch (row_tile) {
  // case (2):
  //   intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (3):
  //   intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (4):
  //   intel_mlperf::_tile_dot_product_16x256<4, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (5):
  //   intel_mlperf::_tile_dot_product_16x256<5, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (6):
  //   intel_mlperf::_tile_dot_product_16x256<6, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (7):
  //   intel_mlperf::_tile_dot_product_16x256<7, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (8):
  //   intel_mlperf::_tile_dot_product_16x256<8, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (9):
  //   intel_mlperf::_tile_dot_product_16x256<9, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (10):
  //   intel_mlperf::_tile_dot_product_16x256<10, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (11):
  //   intel_mlperf::_tile_dot_product_16x256<11, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // case (12):
  //   intel_mlperf::_tile_dot_product_16x256<12, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
  //   break;
  // }

  auto out_ = reinterpret_cast<int8_t (*)[16][64]>(out);
  auto nout_ = reinterpret_cast<int (*)[ldc]>(nout);
  // for (int i = 0; i < row_tile; i++) {
  //   intel_mlperf::compare_naive_output(&nout_[i * 16][0], (int8_t*)out_[i], 16, 64, ldc, 64);
  // } 

  printf("************************ start performance test... **************************\n");
  size_t count = 64000;
  // auto wei400m = new int8_t[6400 * 256 * 256];
  // auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(wei400m);
  auto lstart = Time::now();
  for (size_t i = 0; i < count; i++) {
    // switch (row_tile) {
    // case (2):
    //   intel_mlperf::_tile_dot_product_16x256<2, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (3):
    //   intel_mlperf::_tile_dot_product_16x256<3, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (4):
    //   intel_mlperf::_tile_dot_product_16x256<4, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (5):
    //   intel_mlperf::_tile_dot_product_16x256<5, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (6):
    //   intel_mlperf::_tile_dot_product_16x256<6, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (7):
    //   intel_mlperf::_tile_dot_product_16x256<7, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (8):
    //   intel_mlperf::_tile_dot_product_16x256<8, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (9):
    //   intel_mlperf::_tile_dot_product_16x256<9, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (10):
    //   intel_mlperf::_tile_dot_product_16x256<10, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (11):
    //   intel_mlperf::_tile_dot_product_16x256<11, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // case (12):
    //   intel_mlperf::_tile_dot_product_16x256<12, 16, tile_io>::compute(out, 64, act, wei, bias, scale);
    //   break;
    // }
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

  bool post_op = false;
  float o_scale = 1.5;

  send_data2naive(act, wei, nact, nwei, row_tile, false);
  intel_mlperf::naive_linear(nact, 1024, nwei, 64, nout, ldc, bias, scale, row_tile * 16, post_op, o_scale);
  int8_t p_out[row_tile * 16][ldc];
  printf("****************** accuracy...*********************\n");
  
  accuracy_test_256x256(row_tile, p_out, ldc, act, wei, bias, scale, post_op, o_scale);

  auto p_out_ = reinterpret_cast<int8_t (*)[16][ldc]>(p_out);
  nout_ = reinterpret_cast<decltype(nout_)>(nout);
  // for (int i = 0; i < row_tile; i++) {
  //   intel_mlperf::compare_naive_output(&nout_[i * 16][0], (int8_t*)p_out_[i], 16, 64, ldc, ldc);
  // } 

  printf("************************ start performance test... **************************\n");
  
  int core_num = 1;
  size_t block_num = core_num * 128;
  int counter = core_num * 200;
  int single_loop = block_num / core_num;
  int row_total_num = 50;

  // auto wei400m = new int8_t[block_num * 256 * 256];
  void* wei400m = nullptr;
  posix_memalign(&wei400m, 4096, block_num * 256 * 256);
  auto wei400m_ = reinterpret_cast<int8_t (*)[256 * 256]>(wei400m);

// # pragma omp parallel for
//   for (int i = 0; i < block_num; i++) {
//     set_data_wei(wei400m_[i], bias);
//   }
  memset(wei400m, 1, block_num * 256 * 256);

  std::vector<float> durings;
  durings.emplace_back(0);
 
  for (int i = 2; i < row_total_num; i += 2) {
    int row_tile_ = i;

    void* core_act_ = nullptr;
    posix_memalign(&core_act_, 4096, core_num * row_tile_ * 16 * ldc);
    memset(core_act_, 1, core_num * row_tile_ * 16 * ldc);

    void* core_out_ = nullptr;
    posix_memalign(&core_out_, 4096, core_num * row_tile_ * 16 * ldc);
    lstart = Time::now();
    for (int j = 0; j < counter; j++) {
      performance_test_256x256(row_tile_, core_out_, ldc, core_act_, wei, bias, scale, block_num, core_num);
    }
    lduring = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - lstart).count();
    std::cout << row_tile_ << " linear time : " << (float)lduring / counter / single_loop << " ns durings : " << 
      (float)lduring / counter / single_loop - durings.back() << " ns" << std::endl;
    durings.emplace_back((float)lduring / counter / single_loop);
  }
  
  // for (int i = 0; i < durings.size() - 1; i += 2) {
  //   std::cout << durings[i + 1] - durings[i] << std::endl;
  // }

  // delete[] wei400m;
  free(wei400m);
}

void test_block_gemm(const size_t sl, const size_t input_f, const size_t output_f, const int num_cores) {
  bool has_bias = true;
  bool post_op = false;
  float o_scale = 0.0;
  auto gemm_ = intel_mlperf::i_linear_i8o32(sl, input_f, output_f, has_bias, post_op);

  size_t row_tile = (sl + 15) / 16;
  size_t col_step = output_f / 64;
  size_t col_tile = input_f / 64;

  size_t block_row = input_f / 4;
  size_t block_num = 1;
  const size_t seq_split_len = 512;

  void* weight = nullptr;
  posix_memalign(&weight, 4096, block_num * input_f * output_f);

  float bias[col_step*64];
  float scale = 0.0018;

//# pragma omp parallel for
  for (int i = 0; i < block_num; i++) {
    auto w = reinterpret_cast<int8_t (*)[col_step][4][col_tile][16][64]>(weight);
    intel_mlperf::set_data_wei(w[i], bias, input_f, output_f);
  }
  
  // intel_mlperf::print_2d_matrix<int>((int*)nweight, dim1, dim2, 1024);
  // getchar();

  // intel_mlperf::print_2d_matrix<int>((int*)noutput, dim0, dim2, dim2);
  // getchar();
  // intel_mlperf::print_2d_matrix<int8_t>((int8_t*)output, dim0, dim2, dim2);
  // getchar();
  printf("**************************** start test performance **********************\n");
  int loop_num = 10000;
  auto start = Time::now();
  alignas(64) int8_t input[sl][input_f];
  alignas(64) float output[sl][col_step][64];
  printf("i %d", loop_num);
  for (int i = 0; i < loop_num; i++) {
    for (int j = 0; j < seq_split_len; j++) {
    //# pragma omp parallel for num_threads (num_cores)
      # pragma omp parallel
      {
        auto total_core_num = omp_get_num_threads();
        auto core_id = omp_get_thread_num();
        gemm_.i_linear::tile_dot_product_16x256_shortage(output, input, weight, bias, scale, o_scale, sl, col_step, core_id, total_core_num);
      }
    }
  }
  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  std::cout << sl << " x " << input_f << " x " << output_f << " : " << (float)during / loop_num << " ns " << std::endl;
  delete[] weight;
}

int main(int argc, char* argv[]) {
  int row_tile = 2;
  int is_block = 1;
  int num_cores = 56;
  if (argc == 2) {
    row_tile = std::atoi(argv[1]);
  } else if (argc == 3) {
    num_cores = std::atoi(argv[2]);
  }

  amx_init::amx_init();
  intel_mlperf::Tilecfg().set_config();

  //bool accuracy_mode = true;
  //std::cout << "row_tile : " << row_tile << std::endl;
  test_block_gemm(32, 2048, 4096, 4);

  //intel_mlperf::performance_linear(64, 1024, 4096);
  //intel_mlperf::performance_linear(32, 2048, 4096);
  //intel_mlperf::performance_linear(32, 256, 4096);
  //intel_mlperf::performance_linear_i8o32(32, 1024, 4096);
  //intel_mlperf::performance_linear_i8o32(32, 2048, 4096);
  //intel_mlperf::performance_linear_i8o32(32, 256, 4096);
  //intel_mlperf::performance_gemm(64, 1024, 64);
  
  // intel_mlperf::accuracy_gemm(64, 1024, 64);
  return 0;
}
