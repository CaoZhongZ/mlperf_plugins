#include "preemphasis.hpp"

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "tpps/preemphasis_tpp.hpp"
#include "tpps/reflection_copy_tpp.hpp"

namespace intel_mlperf {
at::Tensor preemphasis(
    const at::Tensor &input, const at::Tensor &length,
    const at::optional<at::Scalar> &coeff, const at::optional<at::Scalar> &pad_size,
    const at::optional<at::Scalar> &part_len, const at::optional<at::Tensor> &index) {
  if (!input.is_contiguous()) {
    throw std::runtime_error("Input should be contiguous.");
  }
  auto in_sz = input.sizes();
  auto batch = in_sz[0];
  auto seq_len = in_sz[1];
  auto pad_size_ = pad_size.value_or(0).toInt();
  if (seq_len < pad_size_ * 2) {
    throw std::runtime_error("seq_len should not be smaller than pad_size * 2.");
  }
  auto part_len_ = seq_len + 2 * pad_size_;
  if (index) {
    part_len_ = part_len.value().toInt();
  }
  auto output = at::empty(
      {batch, part_len_},
      at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto src = input.accessor<float, 2>();
  auto dst = output.accessor<float, 2>();
  auto len = length.accessor<int32_t, 1>();
  auto coeff_ = coeff.value_or(0.97f).toFloat();

  if (coeff_ == 0.0f) {
    output = input;
  } else if (index) {
    auto index_ = index.value().accessor<int32_t, 1>();
#pragma omp parallel for
    for (auto i = 0; i < batch; i++) {
      if (index_[i] < 0) {
        // actual input [0, min(len[i], part_len+idx))
        auto rl = std::min(part_len_ + index_[i], len[i]);
        preemphasis_tpp::ref(&dst[i][-index_[i]], &src[i][0], rl, coeff_, index_[i]);
        // padding [idx, 0)
        reflection_copy_tpp::ref(
            &dst[i][0], &dst[i][pad_size_ + 1], -index_[i], pad_size_ + index_[i]);
      } else {
        // actual input [idx, min(len[i], part_len+idx))
        auto rl = std::min(part_len_, len[i] - index_[i]);
        preemphasis_tpp::ref(&dst[i][pad_size_], &src[i][0], rl, coeff_, index_[i]);
        // possible padding [len[i], min(len[i]+pad_size, part_len+idx))
        auto actual_pad_size = std::min(pad_size_, part_len_ + index_[i] - len[i]);
        reflection_copy_tpp::ref(
            &dst[i][pad_size_ + len[i]], &dst[i][len[i] - 1], actual_pad_size);
      }
      if (pad_size_ > 0) {
        reflection_copy_tpp::ref(&dst[i][0], &dst[i][pad_size_ + 1], pad_size_);
        reflection_copy_tpp::ref(
            &dst[i][pad_size_ + len[i]], &dst[i][len[i] - 1], pad_size_);
      }
      memset(&dst[i][len[i] + 2 * pad_size_], 0, sizeof(float) * (seq_len - len[i]));
    }
  } else {
#pragma omp parallel for
    for (auto i = 0; i < batch; i++) {
      preemphasis_tpp::ref(&dst[i][pad_size_], &src[i][0], len[i], coeff_);
      if (pad_size_ > 0) {
        reflection_copy_tpp::ref(&dst[i][0], &dst[i][pad_size_ + 1], pad_size_);
        reflection_copy_tpp::ref(
            &dst[i][pad_size_ + len[i]], &dst[i][len[i] - 1], pad_size_);
      }
      memset(&dst[i][len[i] + 2 * pad_size_], 0, sizeof(float) * (seq_len - len[i]));
    }
  }

  return output;
}

}  // namespace intel_mlperf
