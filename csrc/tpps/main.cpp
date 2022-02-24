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

template <typename T> void fill_seq(T *t, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    int start = i;
    for (size_t j = 0; j < cols; ++j)
      t[i][j] = start++;
  }
}

void set_data_act(void *a, size_t n_tile)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(seed);
  std::normal_distribution<double> dis(0, 1);
  auto a_ = reinterpret_cast<int8_t (*)>(a);
  size_t elenum = n_tile * 16 * 1024;
  for (int i = 0; i < elenum; i++) {
    a_[i] = static_cast<int8_t>(dis(gen) * 0xf);
    // a_[i] = 0;
  }
}

void set_data_wei(void *w, void* b) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine gen(seed);
  std::normal_distribution<double> dis(0, 1);
  auto w_ = reinterpret_cast<int8_t (*)>(w);
  size_t elenum = 256 * 256;
  for (int i = 0; i < elenum; i++) {
    w_[i] = static_cast<int8_t>(dis(gen) * 0xf);
    // w_[i] = 0;
  }
  auto b_ = reinterpret_cast<float (*)>(b);
  for (int i = 0; i < 64; i++) {
    b_[i] = static_cast<float>(dis(gen) * 0xf);
    // b_[i] = 0;
  }
}

void test_accuracy_linear(int row_tile);

static constexpr size_t cols = 384;
static constexpr size_t rows = 16;
static constexpr int qmax = 127.0;
static constexpr int qmin = -127.0;

int main(int argc, char* argv[]) {
  int row_tile = 2;
  if (argc == 2) {
    row_tile = std::atoi(argv[1]);
  }
  std::cout << "++++++++++++++++++++++++++ row_tile = "<< row_tile << " ++++++++++++++++++++++++++++++++++++" << std::endl;
  
  test_accuracy_linear(row_tile);
  return 0;
}

void naive_linear(void* a, void* b, void* c, void* bias, float scale, int row_tile) {
  auto a_ = reinterpret_cast<int (*)[1024]>(a);
  auto b_ = reinterpret_cast<int (*)[64]>(b);
  auto c_ = reinterpret_cast<int (*)[64]>(c);
  auto bias_ = reinterpret_cast<float (*)>(bias);

  for (int i = 0; i < 16 * row_tile; i++) {
    for (int j = 0; j < 64; j++) {
      c_[i][j] = 0;
    }
  }

  for (int i = 0; i < 16 * row_tile; i++) {
    for (int j = 0; j < 64; j++) {
      for (int k = 0; k < 1024; k++) {
        c_[i][j] += a_[i][k] * b_[k][j];
        // std::cout << i << " " << k << " " << j << " : " << a_[i][k] << " , " << b_[k][j] << " : " << c_[i][j] << std::endl;
      }
      // getchar();
    }
  }

  for (int i = 0; i < 16 * row_tile; i++) {
    for (int j = 0; j < 64; j++) {
      float tem = static_cast<float>(c_[i][j]);
      tem += bias_[j];
      tem *= scale;
      c_[i][j] = static_cast<int>(round(tem));
      c_[i][j] = c_[i][j] < qmax ? c_[i][j] : qmax;
      c_[i][j] = c_[i][j] > qmin ? c_[i][j] : qmin;
    }
  }
}

void send_data2naive(void* a, void* b, void* na, void* nb, int row_tile) {
  auto a_ = reinterpret_cast<int8_t (*)[16][16][64]>(a);
  auto na_ = reinterpret_cast<int (*)[1024]>(na);

  for (int i = 0; i < row_tile; i++) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k++) {
        for (int m = 0; m < 64; m++) {
          na_[i * 16 + k][j * 64 + m] = static_cast<int>(a_[i][j][k][m]);
        }
      }
    }
  }

  auto b_ = reinterpret_cast<int8_t (*)[16][16][64]>(b);
  auto nb_ = reinterpret_cast<int (*)[64]>(nb);

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

void test_accuracy_linear(int row_tile) {
  alignas(64) int8_t act[row_tile * 16 * 1024];
  alignas(64) int8_t wei[256 * 256];
  alignas(64) int nact[row_tile * 16 * 1024];
  alignas(64) int nwei[1024 * 64];
  
  alignas(64) float bias[64];
  alignas(64) int8_t out[row_tile * 16][64];
  alignas(64) int nout[row_tile * 16 * 64];
  float scale = 0.0018;
  set_data_act(act, row_tile);
  set_data_wei(wei, bias);

  send_data2naive(act, wei, nact, nwei, row_tile);
  auto act_ = reinterpret_cast<int8_t (*)[16][16][64]>(act);
  auto nact_ = reinterpret_cast<int (*)[1024]>(nact);
  auto wei_ = reinterpret_cast<int8_t (*)[16][16][64]>(wei);
  auto nwei_ = reinterpret_cast<int (*)[64]>(nwei);

  // for (int i = 0; i < row_tile; i++) {
  //   for (int j = 0; j < 16; j++) {
  //     intel_mlperf::compare_naive_input(&nact_[i * 16][j * 64], (int8_t*)act_[i][j], 16, 64, 1024, 64);
  //     getchar();
  //     std::cout << "input show: int8" << std::endl;
  //     intel_mlperf::print_2d_matrix<int8_t>((int8_t*)act_[i][j], 16, 16, 64);
  //     getchar();
  //     std::cout << "input show: int" << std::endl;
  //     intel_mlperf::print_2d_matrix<int>(&nact_[i * 16][j * 64], 16, 16, 1024);
  //     getchar();
  //   }
  // }
  // printf("compare act done!\n");
  // getchar();

  // for (int i = 0; i < 4; i++) {
  //   for (int j = 0; j < 16; j++) {
  //     intel_mlperf::compare_naive_weight(&nwei_[j * 64][i * 16], (int8_t*)wei_[i][j], 16, 64, 64, 64);
  //     getchar();
  //     std::cout << "weight show: int8_t" << std::endl;
  //     intel_mlperf::print_2d_matrix<int8_t>((int8_t*)wei_[i][j], 16, 64, 64);
  //     getchar();
  //     std::cout << "weight show int" << std::endl;
  //     intel_mlperf::print_2d_matrix<int>((int*)&nwei_[j * 64][i * 16], 64, 16, 64);
  //     getchar();
  //   }
  // }
  // printf("compare wei done!\n");
  // getchar();

  intel_mlperf::amx_init();
  intel_mlperf::Tilecfg().set_config();

  naive_linear(nact, nwei, nout, bias, scale, row_tile);
  switch (row_tile) {
  case (2):
    intel_mlperf::_tile_dot_product_16x256<2, 16>::compute(out, act, wei, bias, scale, 0);
    break;
  case (3):
    intel_mlperf::_tile_dot_product_16x256<3, 16>::compute(out, act, wei, bias, scale, 0);
    break;
  }

  printf("++++++++++++++++compare out and nout : +++++++++++++++++++++\n");
  auto out_ = reinterpret_cast<int8_t (*)[16][64]>(out);
  auto nout_ = reinterpret_cast<int (*)[64]>(nout);
  for (int i = 0; i < row_tile; i++) {
    intel_mlperf::compare_naive_output(&nout_[i * 16][0], (int8_t*)out_[i], 16, 64, 64, 64);
  } 

  // start performance test
  for (int i = 0; i < 100; i++) {
    switch (row_tile) {
    case (2):
      intel_mlperf::_tile_dot_product_16x256<2, 16>::compute(out, act, wei, bias, scale, 0);
      break;
    case (3):
      intel_mlperf::_tile_dot_product_16x256<3, 16>::compute(out, act, wei, bias, scale, 0);
      break;
    }
  }
   
  int count = row_tile == 2 ? 300000 : 200000;
  auto lstart = Time::now();
  for (int i = 0; i < count; i++) {
    switch (row_tile) {
    case (2):
      intel_mlperf::_tile_dot_product_16x256<2, 16>::compute(out, act, wei, bias, scale, 0);
      break;
    case (3):
      intel_mlperf::_tile_dot_product_16x256<3, 16>::compute(out, act, wei, bias, scale, 0);
      break;
    }
  }
  auto lduring =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - lstart)
          .count();
  std::cout << count  << " times tile linear time : "
            << (float)lduring / 1000 / 1000 << " ms " << std::endl;
  std::cout << "single linear time : " << (float)lduring / count << " ns" << std::endl;
}
