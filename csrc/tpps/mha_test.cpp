#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cxxopts.hpp"
#include "i_softmax_tpp.hpp"
#include "i_mha_tpp.hpp"

using Time = std::chrono::high_resolution_clock;

template <typename T> void fill_seq(T *t, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    int start = i;
    for (size_t j = 0; j < cols; ++j)
      t[i][j] = start++ % 16;
  }
}

int main(int argc, char **argv) {
  cxxopts::Options opts("mha_test", "MHA kernel test");
  opts.add_options()
    ("l,seq_len", "Sequence length", cxxopts::value<size_t>()->default_value("384"))
  ;

  auto parsed_opts = opts.parse(argc, argv);
  auto seq_len = parsed_opts["seq_len"].as<size_t>();

  int8_t attention[seq_len][3072];
  fill_seq(attention, seq_len, 3072);

  auto start = Time::now();
  auto during =
      std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start)
          .count();
  std::cout << "100000 times tile softmax time : "
            << (float)during / 1000 / 1000 << " ms " << std::endl;

  return 0;
}
