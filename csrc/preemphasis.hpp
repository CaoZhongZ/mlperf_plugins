#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor preemphasis(const at::Tensor &input,
                       const c10::optional<at::Scalar> &coeff);

}
