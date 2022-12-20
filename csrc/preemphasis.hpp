#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor preemphasis(
    const at::Tensor &input, const at::optional<at::Scalar> &coeff,
    const at::optional<at::Scalar> &pad_size);

}
