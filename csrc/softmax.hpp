#pragma once
#include <torch/torch.h>

namespace intel_mlperf {
at::Tensor i_softmax(
    const at::Tensor& input,
    const at::Tensor& att_mask,
    double M, double oscale);
}
