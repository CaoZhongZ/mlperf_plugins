#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor i_residual_layernorm (
    const at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Scalar& scale_1,
    const at::Scalar& scale_2,
    const at::Scalar& oscale,
    const c10::optional<at::Scalar>& eps);

at::Tensor i_residual_layernorm_ (
    at::Tensor& input1,
    const at::Tensor& input2,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Scalar& scale_1,
    const at::Scalar& scale_2,
    const at::Scalar& oscale,
    const c10::optional<at::Scalar>& eps);

at::Tensor i_layernorm (
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Scalar& oscale,
    const c10::optional<at::Scalar>& eps);

}
