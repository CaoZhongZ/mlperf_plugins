#include "frame_splicing_tpp.hpp"
#include "el_common_intrin.hpp"
#include <cstdlib>
#include <stdexcept>

namespace intel_mlperf {

/*!
 * \brief Input need to be N*F*T layout.
 * \param fi input freq dim index.
 * \param fl input freq dim length.
 * \param tl input time dim length.
 * convert input from:
 * | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 0 | 0 |
 * | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 0 | 0 |
 * to output:
 * | 0  | 3  | 6  | 9  |
 * | 10 | 13 | 16 | 19 |
 * | 1  | 4  | 7  | 0  |
 * | 11 | 14 | 17 | 0  |
 * | 2  | 5  | 8  | 0  |
 * | 12 | 15 | 18 | 0  |
 */
template <>
void frame_splicing_tpp<3>::ref(float *pout, float *pin, int32_t fi, int64_t fl,
                                int64_t tl) {
  const int32_t padded_tl = (tl + 2) / 3;
  // remember the index should be reversed.
  const auto idx_base =
      _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  const auto c3 = _mm512_set1_epi32(3);
  const auto zeros = _mm512_setzero_ps();
  // need to use mullo rather than default mul
  const auto t0 = _mm512_mullo_epi32(idx_base, c3); // idx: 3n
  const auto t1 = t0 + _mm512_set1_epi32(1);        // idx: 3n+1
  const auto t2 = t0 + _mm512_set1_epi32(2);        // idx: 3n+2
  int32_t j = 0;
  for (; j < tl / 48; j++) {
    int32_t base = j * 48 + fi * tl;
    auto base_ = _mm512_set1_epi32(base);

    auto idx_front_ = t0 + base_;
    auto rst_front = _mm512_i32gather_ps(idx_front_, pin, 4);
    // out[j * 16] = int[j * 48 + freq_i * time_len + 3n]
    _mm512_storeu_ps(&pout[j * 16], rst_front);

    auto idx_middle_ = t1 + base_;
    auto rst_middle = _mm512_i32gather_ps(idx_middle_, pin, 4);
    // out[j * 16 + padded_time_len * freq_len]
    // = int[j * 48 + freq_i * time_len + 3n + 1]
    _mm512_storeu_ps(&pout[j * 16 + padded_tl * fl], rst_middle);

    auto idx_back_ = t2 + base_;
    auto rst_back = _mm512_i32gather_ps(idx_back_, pin, 4);
    // out[j * 16 + padded_time_len * freq_len * 2]
    // = int[j * 48 + freq_i * time_len + 3n + 2]
    _mm512_storeu_ps(&pout[j * 16 + padded_tl * fl * 2], rst_back);
  }
  // Tail
  {
    int32_t base = j * 48 + fi * tl;
    int64_t p;
    p = ((fi + 1) * tl - base) / 3;
    __mmask16 mask_front = (1 << p) - 1;
    p = ((fi + 1) * tl - base - 1) / 3;
    __mmask16 mask_middle = (1 << p) - 1;
    p = ((fi + 1) * tl - base - 2) / 3;
    __mmask16 mask_back = (1 << p) - 1;
    auto base_ = _mm512_set1_epi32(base);

    auto idx_front_ = t0 + base_;
    auto rst_front =
        _mm512_mask_i32gather_ps(zeros, mask_front, idx_front_, pin, 4);
    _mm512_mask_storeu_ps(&pout[j * 16], mask_front, rst_front);

    auto idx_middle_ = t1 + base_;
    auto rst_middle =
        _mm512_mask_i32gather_ps(zeros, mask_middle, idx_middle_, pin, 4);
    _mm512_mask_storeu_ps(&pout[j * 16 + padded_tl * fl], mask_middle,
                          rst_middle);

    auto idx_back_ = t2 + base_;
    auto rst_back =
        _mm512_mask_i32gather_ps(zeros, mask_back, idx_back_, pin, 4);
    _mm512_mask_storeu_ps(&pout[j * 16 + padded_tl * fl * 2], mask_back,
                          rst_back);
  }
}

} // namespace intel_mlperf