#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>

#include "amx_mha.hpp"
#include "i_mha_tpp.hpp"
#include "amx_config.hpp"

namespace intel_mlperf {

at::Tensor amx_mha(const at::Tensor &qkv, const at::Tensor &att_mask,
                   const at::Scalar &m1, const at::Scalar &oscale,
                   const at::Scalar &m2) {
  auto qkv_sizes = qkv.sizes();
  assert(qkv_sizes.size() == 3);
  auto bs = qkv_sizes[0];
  auto sl = qkv_sizes[1];
  auto stride = qkv_sizes[2];

  auto qkv_block = stride / 3;
  int head_size = 64;
  int head_num = qkv_block / head_size;

  auto attention = at::empty({bs, sl, qkv_block},
                             at::TensorOptions().dtype<int8_t>().memory_format(
                                 c10::MemoryFormat::Contiguous));

  auto amx_status = amx_init();

  if (!amx_status) {
    printf("amx init failed!\n");
    return attention;
  }

  // create attention tensor
  auto att_ptr = reinterpret_cast<int8_t(*)[sl][head_num][head_size]>(
      attention.data_ptr());
  auto origin_ptr =
      reinterpret_cast<int8_t(*)[sl][3][head_num][head_size]>(qkv.data_ptr());
  auto att_mask_p = reinterpret_cast<int32_t *>(att_mask.data_ptr());

#pragma omp parallel for collapse(2)
  for (int i = 0; i < bs; i++) {         // batch size
    for (int j = 0; j < head_num; j++) { // head num
      auto cur_q_ptr = &origin_ptr[i][0][0][j][0];
      auto cur_a_ptr = &att_ptr[i][0][j][0];

      amx_per_head(cur_q_ptr, stride, cur_a_ptr, qkv_block, sl, m1.toFloat(),
                   oscale.toFloat(), att_mask_p[i], m2.toFloat());
    }
  }
  return attention;
}

} // namespace intel_mlperf
