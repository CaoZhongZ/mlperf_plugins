#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& hx,
    const c10::optional<at::Tensor>& cx,
    const std::vector<std::vector<at::Tensor>> all_weights,
    const c10::optional<int64_t> hidden_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& hx,
    const c10::optional<at::Tensor>& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const c10::optional<at::Tensor>& b_ih,
    const c10::optional<at::Tensor>& b_hh,
    const c10::optional<int64_t> hidden_size);

std::tuple<at::Tensor, at::Tensor> prepack_lstm_weights (
    const at::Tensor& w_ih,
    const at::Tensor& w_hh);

}
