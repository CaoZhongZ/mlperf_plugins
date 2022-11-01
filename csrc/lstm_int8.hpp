#pragma once
#include <torch/torch.h>
#include <vector>

namespace intel_mlperf {

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_int8(
    const at::Tensor& x,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const std::vector<std::vector<at::Tensor>> all_weights,
    const std::vector<at::Scalar>& o_scale,
    const std::vector<at::Scalar>& in_scale,
    const std::vector<at::Scalar>& out_scale,
    const bool flag,
    const bool skip_quant_y);
    
std::tuple<std::vector<at::Tensor>, at::Tensor, at::Tensor> lstm_rnnt_layer(
    const std::vector<at::Tensor>& x,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& b_ih,
    const at::Tensor& b_hh,
    const c10::optional<at::Scalar>& o_scale,
    const c10::optional<at::Scalar>& in_scale,
    const c10::optional<at::Scalar>& out_scale,
    const bool flag,
    const bool skip_quant_y);

std::tuple<std::vector<at::Tensor>, at::Tensor, at::Tensor> lstm_rnnt_cell(
    const std::vector<at::Tensor>& x,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& b_ih,
    const at::Tensor& b_hh,
    const c10::optional<at::Scalar>& o_scale,
    const c10::optional<at::Scalar>& in_scale,
    const c10::optional<at::Scalar>& out_scale,
    const bool skip_quant_y);

// std::tuple<std::vector<at::Tensor>, at::Tensor, at::Tensor> lstm_single_cell(
//     const std::vector<at::Tensor>& x,
//     const at::Tensor& hx,
//     const at::Tensor& cx,
//     const at::Tensor& w_ih,
//     const at::Tensor& w_hh,
//     const at::Tensor& b_ih,
//     const at::Tensor& b_hh,
//     const c10::optional<at::Scalar>& o_scale,
//     const c10::optional<at::Scalar>& in_scale,
//     const c10::optional<at::Scalar>& out_scale,
//     const bool skip_quant_y);

}