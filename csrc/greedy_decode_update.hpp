#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

bool greedy_decode_update(
    const at::Tensor &symbols, at::Tensor &symbols_added, at::Tensor &res,
    at::Tensor &res_idx, at::Tensor &time_idx, const at::Tensor &f_lens,
    at::Tensor &pred_g,
    at::Tensor &f, at::Tensor &fi, at::Tensor &pred_hg, at::Tensor &pred_cg,
    at::Tensor &pred_state_hg, at::Tensor &pred_state_cg);

}
