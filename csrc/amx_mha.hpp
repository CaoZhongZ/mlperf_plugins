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

#define MAX_SL 384
#define MAX_TILE_ROW 16
#define MAX_TILE_COLSB 64

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

static int8_t k_buffer[16 * MAX_SL * 4];

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
        
        nq_block = (sl / MAX_TILE_ROW >= 2) ? 2 : 1;
        nk_block = nq_block;
        q_block = MAX_TILE_ROW;
        k_block = MAX_TILE_ROW;
        q_colsb = MAX_TILE_COLSB;
        k_colsb = MAX_TILE_COLSB;

        a_r_block = q_block;
        a_c_block = k_colsb / typesize_A;

        q_tail = sl_ % q_block;
        is_q_tail = (q_tail == 0) ? false : true;

        // only q has tail, k's tail must be 16*64
        nbq_row = is_q_tail ? (sl / MAX_TILE_ROW + 1) : (sl / MAX_TILE_ROW);
        nbk_col = nbq_row;

        att_stride_ = sl_;

        strides_ = {sl_ * qkv_stride_, qkv_stride_, 1};
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
    int a_r_block;
    int a_c_block;
    bool is_q_tail;
    int q_tail;
    int typesize_Q;
    int typesize_K;
    int typesize_A;

    int nbq_row;
    int nbk_col;

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

inline status_t configure_tile(struct tilecfg *cfg, int ntile, int row, int colsb)
{
    if (row <= 0 && colsb <= 0 && row > 16 && colsb > 64 && ntile > 7){
        return status_t::failed;
    }
    cfg->tile_rows[ntile] = row;
    cfg->tile_colsb[ntile] = colsb;
    return status_t::success;
}

status_t reorder_k_to_buffer(const int8_t* k_ptr, int row, int col, int stride);

status_t mha_init_tile(struct tilecfg *cfg, MHA_desc& mhad);

status_t amx_qk_gemm(const int8_t* q_ptr, const int8_t* k_ptr, int* a_ptr, MHA_desc& mhad);

at::Tensor amx_mha(
    const at::Tensor& qkv,
    const at::Tensor& attpro,
    const at::Scalar& M1,
    const at::Scalar& oscale 
);

}