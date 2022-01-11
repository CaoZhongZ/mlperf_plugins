#pragma once
#include <asm/prctl.h>        /* Definition of ARCH_* constants */
#include <sys/syscall.h>      /* Definition of SYS_* constants */
#include <unistd.h>

#include <torch/torch.h>

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define TMM0	0
#define TMM1	1
#define TMM2	2
#define TMM3	3
#define TMM4	4
#define TMM5	5
#define TMM6	6
#define TMM7	7

namespace intel_mlperf {

enum class status_t {success, failed};

static struct tilecfg {
	uint8_t palette;	    /* byte 0 */
	uint8_t start_row;	    /* byte 1 */
	char rsvd1[14];		    /* bytes 2-15 */
	uint16_t tile_colsb[8];	/* bytes 16-31 */
	char rsvd2[16];		    /* bytes 32-47 */
	uint8_t tile_rows[8];	/* bytes 48-55 */
	char rsvd3[8];		    /* bytes 56-63 */
} __attribute__((packed)) tilecfg;

class MHA_desc {
public:
    MHA_desc(int sl) : sl_(sl) {
        nq_block = 2;
        nk_block = 2;
        q_block = 16;
        k_block = 16;
        q_colsb = 64;
        k_colsb = 64;
        q_tail = sl_ % q_block;
        is_q_tail = (q_tail == 0) ? false : true;

        // note: typesize is for s8s8s32 only
        typesize_Q = 1;
        typesize_K = 1;
        typesize_Q = 4;
    };
    
    int get_q_ntile(int nb) {
        return nb + 4;
    }

    int get_k_ntile(int nb) {
        return nb + 6;
    }

    int get_a_ntile(int nb_q, int nb_k) {
        return nb_q * this->nq_block + nb_k;
    }

    status_t init() {return status_t::success;}; // TODO: which to init
    
    virtual ~MHA_desc() = default;

    int sl_;
    int nq_block;
    int nk_block;
    int q_block;
    int q_colsb;
    int k_block;
    int k_colsb;
    bool is_q_tail;
    int q_tail;
    int typesize_Q;
    int typesize_K;
    int typesize_A;
};

inline bool amx_init() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    return true;
}

inline status_t configure_tile(struct tilecfg *cfg, int ntile, int row, int colsb)
{
    if (row <= 0 && colsb <= 0 && row > 16 && colsb > 64 && ntile > 7){
        return status_t::failed;
    }
    cfg->tile_rows[ntile] = row;
    cfg->tile_colsb[ntile] = colsb;
    return status_t::success;
}

status_t mha_init_tile(struct tilecfg *cfg, MHA_desc& mhad);

at::Tensor amx_mha(
    const at::Tensor& qkv,
    const at::Tensor& attpro,
    const at::Scalar& M1,
    const at::Scalar& oscale 
);

}