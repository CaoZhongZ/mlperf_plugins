#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor i_gelu (
    const at::Tensor& input,
    double M,
    double oscale);
}
