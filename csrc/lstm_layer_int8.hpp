#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_int8(
    const at::Tensor &x, const at::Tensor &hx, const at::Tensor &cx,
    const at::Tensor &w_ih, const at::Tensor &w_hh, const at::Tensor &b_ih,
    const at::Tensor &b_hh, const c10::optional<at::Scalar> &rb_scale,
    const c10::optional<at::Scalar> &i_scale,
    const c10::optional<at::Scalar> &o_scale, const bool skip_quant_y);

}
