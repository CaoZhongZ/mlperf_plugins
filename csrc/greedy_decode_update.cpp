#include "greedy_decode_update.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "tpps/greedy_decode_update_tpp.hpp"

const static size_t kBatchNum = 16;

namespace intel_mlperf {
typedef unsigned short __bfloat16;
bool greedy_decode_update(
    const at::Tensor &symbols,
    at::Tensor &symbols_added,
    at::Tensor &res,
    at::Tensor &res_idx,
    const at::Tensor &f,
    const at::Tensor &f_lens,
    at::Tensor &time_idx,
    at::Tensor &fi,
    at::Tensor &pre_g,
    const std::vector<at::Tensor> &pre_hg,
    const std::vector<at::Tensor> &pre_cg,
    const std::vector<at::Tensor> &hg,
    const std::vector<at::Tensor> &cg) {
  auto res_sz = res.sizes();
  auto batch_size = res_sz[0];
  auto seq_len = res_sz[1];
  auto trans_hidden_size = f.sizes()[2];
  auto pred_hidden_size = hg[0].sizes()[1];

  auto symbols_ = symbols.accessor<int64_t, 1>();
  auto symbols_added_ = symbols_added.accessor<int32_t, 1>();
  auto res_ = res.accessor<int32_t, 2>();
  auto res_idx_ = res_idx.accessor<int32_t, 1>();
  auto time_idx_ = time_idx.accessor<int32_t, 1>();
  auto f_lens_ = f_lens.accessor<int32_t, 1>();
  auto pre_g_ = pre_g.accessor<int32_t, 2>();
  //auto finish_t_ = finish_t.accessor<bool, 1>();

  size_t loop_num = (batch_size - 1) / kBatchNum + 1;

  unsigned short update_g[loop_num];
  unsigned short finish[loop_num];

  bool flag[loop_num];
#pragma omp parallel for
  for (auto i = 0; i < loop_num; i++) {
    // simd part
    auto start = i * kBatchNum;
    auto batch = kBatchNum;
    if (batch_size - start < kBatchNum) batch = batch_size - start;
    greedy_decode_update_tpp::update_mask(
        &symbols_[start], &symbols_added_[start], &res_[start][0],
        &res_idx_[start], &time_idx_[start], &f_lens_[start],
        &pre_g_[0][start], seq_len, batch, update_g[i], finish[i]);

    // data copy part
    auto pin_f = reinterpret_cast<__bfloat16 (*)[batch_size][trans_hidden_size]>(f.data_ptr());
    auto pin_hg_0 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(hg[0].data_ptr());
    auto pin_hg_1 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(hg[1].data_ptr());
    auto pin_cg_0 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(cg[0].data_ptr());
    auto pin_cg_1 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(cg[1].data_ptr());
    auto pout_fi = reinterpret_cast<__bfloat16 (*)[trans_hidden_size]>(fi.data_ptr());
    auto pout_pre_hg_0 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(pre_hg[0].data_ptr());
    auto pout_pre_hg_1 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(pre_hg[1].data_ptr());
    auto pout_pre_cg_0 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(pre_cg[0].data_ptr());
    auto pout_pre_cg_1 = reinterpret_cast<__bfloat16 (*)[pred_hidden_size]>(pre_cg[1].data_ptr());
    // check whether finished
    // flag is false only when finish is all true.
    if (batch_size - start < kBatchNum) {
      unsigned short temp = (~finish[i]) << (kBatchNum - batch);
      flag[i] = temp;
    } else {
      unsigned short temp = ~finish[i];
      flag[i] = temp;
    }
    for (auto j = 0; j < batch; j++) {
      //finish_t[start + j] = finish[i] & (1 << j);
      if (~finish[i] & (1 << j)) {
        auto idx = time_idx_[start + j];
        memcpy(pout_fi[start + j], pin_f[idx][start + j], sizeof(pout_fi[start + j]));
      }
      //updateg_t[start + j] = update_g[i] & (1 << j);
      if (update_g[i] & (1 << j)) {
        auto out_byte = sizeof(pin_hg_0[start + j]);
        memcpy(pout_pre_hg_0[start + j], pin_hg_0[start + j], out_byte);
        memcpy(pout_pre_hg_1[start + j], pin_hg_1[start + j], out_byte);
        memcpy(pout_pre_cg_0[start + j], pin_cg_0[start + j], out_byte);
        memcpy(pout_pre_cg_1[start + j], pin_cg_1[start + j], out_byte);
      }
    }
  }
  bool rst = false;
  for (auto i = 0; i < loop_num; i++) {
    rst |= flag[i];
  }
  return !rst;
}

}  // namespace intel_mlperf
