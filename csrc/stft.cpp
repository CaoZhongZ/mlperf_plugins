#include "stft.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <immintrin.h>

namespace intel_mlperf {
inline void helper(float *pout, float *pin, float *win, int64_t rl) {
  int64_t d = 0;
  auto c0 = _mm512_set1_ps(0.0);
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto a = _mm512_loadu_ps(&pin[d]);
    auto b = _mm512_loadu_ps(&win[d]);
    _mm512_storeu_ps(&pout[d], _mm512_mul_ps(a, b));
  }

  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1 << rem) - 1;
    auto a = _mm512_mask_loadu_ps(c0, k, &pin[d]);
    auto b = _mm512_mask_loadu_ps(c0, k, &win[d]);
    _mm512_mask_storeu_ps(&pout[d], k, _mm512_mul_ps(a, b));
  }
}

at::Tensor stft(
    const at::Tensor &input, const at::Tensor &window, const at::Scalar &n_fft,
    const at::Scalar &hop_length, const at::Scalar &win_length) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  auto batch = in_sz[0];
  auto seq_len = in_sz[1];
  auto n_fft_ = n_fft.toLong();
  auto hop_length_ = hop_length.toLong();
  auto win_length_ = win_length.toLong();
  auto n_frames = 1 + (seq_len - n_fft_) / hop_length_;
  auto output = at::empty(
      {batch, n_frames, n_fft_},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto src = input.accessor<float, 2>();
  auto dst = output.accessor<float, 3>();
  auto window_ = window.accessor<float, 1>();
  auto left = (n_fft_ - win_length_) / 2;

#pragma omp parallel for
  for (auto i = 0; i < batch; ++i) {
    for (auto j = 0; j < n_frames; ++j) {
      memset(&dst[i][j][0], 0, left * sizeof(float));
      helper(
          &dst[i][j][left], &src[i][j * hop_length_ + left], &window_[left],
          win_length_);
      memset(&dst[i][j][left + win_length_], 0, left * sizeof(float));
    }
  }

  // return output;
  output = at::_fft_r2c(output, 2, 0, true);
  output.transpose_(1, 2);
  return at::view_as_real(output);
}

}  // namespace intel_mlperf
