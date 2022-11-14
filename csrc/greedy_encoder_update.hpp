#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

bool greedy_encoder_update(
    const at::Tensor &symbols, at::Tensor &symbols_added, at::Tensor &res,
    at::Tensor &res_idx, at::Tensor &time_idx, const at::Tensor &f_lens,
    at::Tensor &pred_g, at::Tensor &finish_t, at::Tensor &updateg_t);

}
