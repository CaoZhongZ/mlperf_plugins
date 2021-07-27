#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor i_residual_layernorm (
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double scale_1, double scale_2,
    double oscale, c10::optional<double> eps);

at::Tensor i_layernorm (
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double oscale, c10::optional<double> eps);
}
