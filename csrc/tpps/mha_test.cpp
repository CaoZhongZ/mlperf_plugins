#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cxxopts.hpp"
#include "i_softmax_tpp.hpp"
#include "i_mha_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

template <typename T> void fill_seq(T *t, size_t rows, size_t cols) {
  int period = 5;
  int start = 0;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (-- period == 0) {
        start += 1;
        period = 4;
      }
      t[i][j] = start % 42;
    }
  }
}

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_SUPP 0x1021
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

#define ARCH_MAP_VDSO_X32 0x2001
#define ARCH_MAP_VDSO_32 0x2002
#define ARCH_MAP_VDSO_64 0x2003

inline bool amx_init() {
  unsigned long bitmask = 0;
  long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (0 != status)
    return false;
  if (bitmask & XFEATURE_MASK_XTILEDATA)
    return true;

  status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (0 != status)
    return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not
                  // allowed
  status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

  // XFEATURE_XTILEDATA setup is failed, can't use TMUL
  if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA))
    return false;

  // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
  return true;
}

// helpers to print a tile of array
void _i8(const void *arr, size_t rs, size_t rt, size_t cs, size_t ct,
    size_t stride) {
  auto t = reinterpret_cast<const int8_t (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int i = rs; i < rt; ++i) {
    for (int j = cs; j < ct; ++j) {
      std::cout<<(int)t[i][j];
      if (j == ct -1)
        std::cout<<";";
      else
        std::cout<<",";
    }
    std::cout<<std::endl;
  }
}

void _i8_row(const void *arr, size_t row, size_t col, size_t stride) {
  auto t = reinterpret_cast<const int8_t (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int j = 0; j < col; ++j) {
    std::cout<<(int)t[row][j]<<",";
  }
  std::cout<<std::endl;
}

void _i8_col(const void *arr, size_t row, size_t col, size_t stride) {
  auto t = reinterpret_cast<const int8_t (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int i = 0; i < row; ++i) {
    std::cout<<(int)t[i][col]<<";";
    std::cout<<std::endl;
  }
}

void _i32(const void *arr, size_t row, size_t col, size_t stride) {
  auto t = reinterpret_cast<const int (*)[stride]>(arr);
  std::cout<<"tile @"<<arr<<":"<<std::endl;
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      std::cout<<(int)t[i][j];
      if (j == col -1)
        std::cout<<";";
      else
        std::cout<<",";
    }
    std::cout<<std::endl;
  }
}

int main(int argc, char **argv) {
  cxxopts::Options opts("mha_test", "MHA kernel test");
  opts.add_options()
    ("l,seq_len", "Sequence length", cxxopts::value<size_t>()->default_value("384"))
  ;

  amx_init();

  auto parsed_opts = opts.parse(argc, argv);
  auto seq_len = parsed_opts["seq_len"].as<size_t>();

  int8_t attention[seq_len][3072];
  fill_seq(attention, seq_len, 3072);

  int8_t result[seq_len][1024];
  memset(result, 0, sizeof(result));

  // Stepping in 64 and do all the columns
  auto att = reinterpret_cast<int8_t (*)[64]>(attention);
  auto res = reinterpret_cast<int8_t (*)[64]>(result);
  intel_mlperf::i_amx_mha_tpp mha(seq_len, 64);

  auto start = Time::now();
  for (int i = 0; i < 16; ++ i) {
    mha.compute_head(res[i], att[i], 3072, 0.0001, 8200, 0.0001);
  }
  auto during =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
          .count();
  std::cout << "100000 times tile softmax time : "
            << (float)during / 1000 / 1000 << " ms " << std::endl;

  return 0;
}
