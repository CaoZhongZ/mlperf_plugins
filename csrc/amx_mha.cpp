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

  auto in_prt =
      reinterpret_cast<int8_t(*)[sl * stride]>(qkv.data_ptr());
  auto out_ptr = reinterpret_cast<int8_t(*)[sl * qkv_block]>(
      attention.data_ptr());
  auto att_mask_p = reinterpret_cast<int32_t *>(att_mask.data_ptr());
  i_amx_mha_tpp compute(sl, head_size);

#pragma omp parallel for collapse(2)
  for (int b = 0; b < bs; b++) {         // batch size
    for (int h = 0; h < head_num; h++) { // head num
      auto att = reinterpret_cast<int8_t (*)[64]>(in_ptr[b]);
      auto res = reinterpret_cast<int8_t (*)[64]>(out_ptr[b]);

      compute.compute_head(
          res[h], att[h], sl, m1.toFloat(), oscale.toFloat(), m2.toFloat());
    }
  }
  return attention;
}

} // namespace intel_mlperf
