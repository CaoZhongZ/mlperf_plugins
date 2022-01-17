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

// #define MAX_SL 384
// #define MAX_TILE_ROW 16
// #define MAX_TILE_COLSB 64

namespace intel_mlperf {

static constexpr int max_sl = 384;
static constexpr int max_tile_row = 16;
static constexpr int max_tile_colsb = 64;
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


// Do we need tile buffer?
struct tilebuffer {
    int8_t a_buffer[1024];
    int8_t b_buffer[1024];
};

class MHA_desc {
public:
    MHA_desc(int bs, int sl, int stride, int head_num, int head_size) 
    : batch_size_(bs),
      sl_(sl),
      head_num_(head_num),
      head_size_(head_size),
      qkv_stride_(stride) {

        // note: typesize is for s8s8s32 only
        typesize_Q = 1;
        typesize_K = 1;
        typesize_A = 4;
        
        nq_block = (sl_ / max_tile_row >= 1) ? 2 : 1;
        nk_block = nq_block;
        q_block = max_tile_row;
        k_block = max_tile_row;
        q_colsb = max_tile_colsb;
        k_colsb = max_tile_colsb;

        a_r_block = q_block;
        a_c_block = k_colsb / typesize_A;

        q_tail = sl_ % q_block;
        is_q_tail = (q_tail == 0) ? false : true;

        // only q has tail, k's tail must be 16*64
        nbq_row = (sl_ + 15) / max_tile_row;
        nbk_col = nbq_row;

        nBlock = nbq_row / 2;

        sl_pad = max_tile_row * nbq_row;

        att_stride_ = max_tile_row * nbq_row;

        strides_ = {sl_ * qkv_stride_, qkv_stride_, 1};
    };
    
    int get_q_ntile(int nb) {
        return nq_block == 2 ? nb + 4 : nb + 5;
    }

    int get_k_ntile(int nb) {
        return nb + 6;
    }

    int get_a_ntile(int nb_q, int nb_k) {
        return nq_block == 2 ? nb_q * this->nq_block + nb_k : 2;
    }

    status_t init() {
        return status_t::success;
        }; // TODO: which to init
    
    virtual ~MHA_desc() = default;

    int sl_;
    int sl_pad;
    int nq_block;
    int nk_block;
    int q_block;
    int q_colsb;
    int k_block;
    int k_colsb;
    int a_r_block;
    int a_c_block;
    bool is_q_tail;
    int q_tail;
    int typesize_Q;
    int typesize_K;
    int typesize_A;

    int nbq_row;
    int nbk_col;
    int nBlock;

    int head_num_;
    int head_size_;
    int batch_size_;
    int qkv_stride_;
    int att_stride_;
    std::vector<int64_t> strides_;
    // TODO: add k tail ?
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


at::Tensor amx_mha(
    const at::Tensor& qkv,
    const at::Tensor& att_mask,
    const at::Scalar& m1,
    const at::Scalar& oscale,
    const at::Scalar& m2
);

}