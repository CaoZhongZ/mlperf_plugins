#pragma once

#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>

namespace intel_mlperf {

struct i_amx_mha_tpp {
  i_amx_mha_tpp(size_t seq_len, size_t head_len);

  template <int row_tile, int col_tile>
  void compute_block(void *C, const void* Q, const void* K, const void* V,
      size_t ld_att, float M, float oscale, float M2);

  void compute_head(void *C, const void *ATT, int ld_att, float M, float oscale,
      float M2);

private:
  typedef void (i_amx_mha_tpp::* compute_block_t) (
      void*, const void*, const void*, const void*, size_t,
      float, float, float);

  const size_t seq_len_;
  const size_t head_len_;
  
  const size_t sl_p16_;
  const size_t sl_p64_;
  
  const size_t overlap_;
  
  size_t loop_block_;
  size_t loop_tail_;

  static const compute_block_t compute_block_tbl_ [2][25];

  compute_block_t compute_block_;
  compute_block_t compute_tail_;
};

} // namespace intel_mlperf
