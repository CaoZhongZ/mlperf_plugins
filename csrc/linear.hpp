#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<double> scale,
    const c10::optional<int64_t> zero);

at::Tensor linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<double> M,
    const c10::optional<double> scale,
    const c10::optional<int64_t> zero);

at::Tensor prepack_linear_weight(
    const at::Tensor& weight);

at::Tensor baddbmm_out_(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    double beta,
    double alpha);

at::Tensor matmul_out_(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    c10::optional<double> oscale,
    c10::optional<int64_t> zero);

at::Tensor reorder_test(const at::Tensor& weight);
}
