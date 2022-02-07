#pragma once

#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>

namespace intel_mlperf {

class Tilecfg {
public:
  void set_config(bool reconfig) const {
    if (reconfig) {
      _tile_release();
      _tile_loadconfig(&cfg);
    }
  }

  void set_config() const {
    _tile_release();
    _tile_loadconfig(&cfg);
  }

  Tilecfg() {
    memset(&cfg, 0, sizeof(cfg));

    for (int i = 0; i < num_valid; i++) {
      cfg.tile_rows[i] = 16;
      cfg.tile_colsb[i] = 64;
    }
    cfg.palette = 1;
  }

  Tilecfg(int k) {
    memset(&cfg, 0, sizeof(cfg));

    // static allocation
    // A: 16 x k
    cfg.tile_rows[4] = 16;
    cfg.tile_colsb[4] = k * sizeof(int);
    cfg.tile_rows[5] = 16;
    cfg.tile_colsb[5] = k * sizeof(int);
    // B: k x 16
    cfg.tile_rows[6] = k;
    cfg.tile_colsb[6] = 16 * sizeof(int);
    cfg.tile_rows[7] = k;
    cfg.tile_colsb[7] = 16 * sizeof(int);
    // C: 16 x 16
    for (int i = 0; i < 4; i++) {
      cfg.tile_rows[i] = 16;
      cfg.tile_colsb[i] = 16 * sizeof(int);
    }
    cfg.palette = 1;
  }

  static constexpr int num_valid = 8;
  struct cfg {
    uint8_t palette;        /* byte 0 */
    uint8_t start_row;      /* byte 1 */
    char rsvd1[14];         /* bytes 2-15 */
    uint16_t tile_colsb[8]; /* bytes 16-31 */
    char rsvd2[16];         /* bytes 32-47 */
    uint8_t tile_rows[8];   /* bytes 48-55 */
    char rsvd3[8];          /* bytes 56-63 */
  } __attribute__((packed)) cfg;
};

struct i_amx_mha_tpp {
  i_amx_mha_tpp(size_t seq_len, size_t head_len);

  template <int row_tile, int col_tile>
  void compute_block(void *C, const void* Q, const void* K, const void* V,
      size_t ld_att, float M, float oscale, float M2, int overlap);

  void compute_head(void *C, const void *ATT, int ld_att, float M, float oscale,
      float M2);

private:
  typedef void (i_amx_mha_tpp::* compute_block_t) (
      void*, const void*, const void*, const void*, size_t,
      float, float, float, int);

  const size_t seq_len_;
  const size_t head_len_;
  
  const size_t sl_p16_;
  const size_t sl_p64_;
  
  const size_t overlap_;
  
  size_t loop_block_;
  size_t loop_tail_;

  static const compute_block_t compute_block_tbl_ [2][25];

  compute_block_t compute_block_;
  compute_block_t compute_tail_;
};

} // namespace intel_mlperf
