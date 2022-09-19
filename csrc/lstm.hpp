#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

std::vector<at::Tensor> lstm(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& hx,
    const c10::optional<at::Tensor>& cx,
    const std::vector<std::vector<at::Tensor>> all_weights);

std::vector<at::Tensor> lstm_layer(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& hx,
    const c10::optional<at::Tensor>& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const c10::optional<at::Tensor>& b_ih,
    const c10::optional<at::Tensor>& b_hh);

std::vector<at::Tensor> prepack_lstm_weights (
    const at::Tensor& x,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const c10::optional<at::Tensor>& bias);

}
