#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <asm/prctl.h>        /* Definition of ARCH_* constants */
#include <sys/syscall.h>      /* Definition of SYS_* constants */
#include <unistd.h>

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_SUPP 0x1021
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

#define ARCH_MAP_VDSO_X32 0x2001
#define ARCH_MAP_VDSO_32 0x2002
#define ARCH_MAP_VDSO_64 0x2003

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

namespace intel_mlperf {

bool amx_init() {
  unsigned long bitmask = 0;
  long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (0 != status)
    return false;
  if (bitmask & XFEATURE_MASK_XTILEDATA)
    return true;

  status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (0 != status)
    return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not
                  // allowed
  status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

  // XFEATURE_XTILEDATA setup is failed, can't use TMUL
  if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA))
    return false;

  // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
  return true;
}

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

private:
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

}