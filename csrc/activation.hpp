#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor i_gelu (
    const at::Tensor& input, const at::Scalar& M, const at::Scalar& oscale);
}
