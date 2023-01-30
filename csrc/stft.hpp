#pragma once
#include <torch/torch.h>

namespace intel_mlperf {

at::Tensor stft(
    const at::Tensor &input, const at::Tensor &window, const at::Scalar &n_fft,
    const at::Scalar &hop_length, const at::Scalar &win_length);

}
