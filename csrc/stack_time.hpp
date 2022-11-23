#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

std::tuple<at::Tensor, at::Tensor> stack_time(
    const at::Tensor &input,
    const at::Tensor &input_lens,
    const at::Scalar &factor);

}
