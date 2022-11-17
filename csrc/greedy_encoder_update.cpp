#include "greedy_encoder_update.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "tpps/greedy_encoder_update_tpp.hpp"

const static size_t kBatchNum = 16;

namespace intel_mlperf {
typedef unsigned short __bfloat16;
bool greedy_encoder_update(
    const at::Tensor &symbols, at::Tensor &symbols_added, at::Tensor &res,
    at::Tensor &res_idx, at::Tensor &time_idx, const at::Tensor &f_lens,
    at::Tensor &pred_g, at::Tensor &finish_t, at::Tensor &updateg_t,
    at::Tensor &f, at::Tensor &fi, at::Tensor &pred_hg, at::Tensor &pred_cg,
    at::Tensor &pred_state_hg, at::Tensor &pred_state_cg) {
  auto res_sz = res.sizes();
  auto batch_size = res_sz[0];
  auto seq_len = res_sz[1];
  auto hidden_size = f.sizes()[2];
  auto pred_hidden_size = pred_state_hg.sizes()[2];

  auto symbols_ = symbols.accessor<int64_t, 1>();
  auto symbols_added_ = symbols_added.accessor<int32_t, 1>();
  auto res_ = res.accessor<int32_t, 2>();
  auto res_idx_ = res_idx.accessor<int32_t, 1>();
  auto time_idx_ = time_idx.accessor<int32_t, 1>();
  auto f_lens_ = f_lens.accessor<int32_t, 1>();
  auto pred_g_ = pred_g.accessor<int32_t, 2>();
  auto finish_t_ = finish_t.accessor<bool, 1>();
  auto updateg_t_ = updateg_t.accessor<bool, 1>();

  size_t loop_num = (batch_size - 1) / kBatchNum + 1;

  int16_t update_g[loop_num];
  int16_t finish[loop_num];

  // simd part
#pragma omp parallel for
  for (auto i = 0; i < loop_num; i++) {
    auto start = i * kBatchNum;
    auto batch = kBatchNum;
    if (batch_size - start < kBatchNum) batch = batch_size - start;
    greedy_encoder_update_tpp::update_mask(
        &symbols_[start], &symbols_added_[start], &res_[start][0],
        &res_idx_[start], &time_idx_[start], &f_lens_[start],
        &pred_g_[0][start], seq_len, batch, update_g[i], finish[i]);
  }

  // data copy part
  auto pin_f = reinterpret_cast<__bfloat16 (*)[batch_size][hidden_size]>(f.data_ptr());
  auto pin_pred_state_hg = reinterpret_cast<__bfloat16 (*)[batch_size][pred_hidden_size]>(pred_state_hg.data_ptr());
  auto pin_pred_state_cg = reinterpret_cast<__bfloat16 (*)[batch_size][pred_hidden_size]>(pred_state_cg.data_ptr());
  auto pout_fi = reinterpret_cast<__bfloat16 (*)[hidden_size]>(fi.data_ptr());
  auto pout_pred_hg = reinterpret_cast<__bfloat16 (*)[batch_size][pred_hidden_size]>(pred_hg.data_ptr());
  auto pout_pred_cg = reinterpret_cast<__bfloat16 (*)[batch_size][pred_hidden_size]>(pred_cg.data_ptr());
  // check whether finished
  bool flag = false;
  for (auto i = 0; i < loop_num; i++) {
    // flag is false only when finish is all true.
    flag |= ~finish[i];
    auto start = i * kBatchNum;
    auto batch = kBatchNum;
    if (batch_size - start < kBatchNum) batch = batch_size - start;
    for (auto j = 0; j < batch; j++) {
      finish_t[start + j] = finish[i] & (1 << j);
      if(~(finish[i] & (1 << j))){
        auto idx = time_idx[start + j].item().toInt();
        memcpy(pout_fi[start + j], pin_f[idx][start + j], sizeof(pout_fi[start + j]));
      }
      updateg_t[start + j] = update_g[i] & (1 << j);
      if(update_g[i] & (1 << j)){
        auto out_byte = sizeof(pout_pred_hg[0][start + j]);
        memcpy(pout_pred_hg[0][start + j], pin_pred_state_hg[0][start + j], out_byte);
        memcpy(pout_pred_hg[1][start + j], pin_pred_state_hg[1][start + j], out_byte);
        memcpy(pout_pred_cg[0][start + j], pin_pred_state_cg[0][start + j], out_byte);
        memcpy(pout_pred_cg[1][start + j], pin_pred_state_cg[1][start + j], out_byte);
      }
    }
  }
  return !flag;
}

}  // namespace intel_mlperf
