#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>

#include "amx_mha.hpp"
#include "i_mha_tpp.hpp"

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

namespace intel_mlperf {

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
