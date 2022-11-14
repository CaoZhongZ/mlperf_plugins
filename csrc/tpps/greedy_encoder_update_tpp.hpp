#pragma once
#include <cstdlib>

namespace intel_mlperf {

class greedy_encoder_update_tpp {
public:
  static const int kBlank = 28;
  static const int kMaxSymbolsPerStep = 30;

  static void update_mask(
      int64_t* symbols, int32_t* symbols_added, int32_t* res, int32_t* res_idx,
      int32_t* time_idx, int32_t* f_lens, int32_t* pred_g, size_t seq_len,
      size_t batch, int16_t& update_g, int16_t& finish);
};

}  // namespace intel_mlperf
