#include "tanh_tpp.hpp"
#include "el_common_intrin.hpp"
#include <cstdlib>

namespace intel_mlperf {
template <int vec_length>
void tanh_tpp<vec_length>::ref(void *out, void *in, int64_t nelem) {
    constexpr int unroll = 4;
    auto constexpr b = tanh_fp16<vec_length, unroll>::batch;
    auto n_batch = nelem / b;

    auto pin = reinterpret_cast<float *>(in);
    auto pout = reinterpret_cast<_Float16 *>(out);

    for (int p = 0; p < n_batch; ++p, pout += b, pin += b) {
    tanh_fp16<vec_length,unroll>::run(pout,pin);
    }
}

template <int vec_length>
void tanh_tpp<vec_length>::ref_(void *out, void *in, int64_t nelem) {
    constexpr int unroll = 4;
    auto constexpr b = tanh_fp16<vec_length, unroll>::batch;
    auto n_batch = nelem / b;

    auto pin = reinterpret_cast<_Float16 *>(in);
    auto pout = reinterpret_cast<_Float16 *>(out);

    for (int p = 0; p < n_batch; ++p, pout += b, pin += b) {
    tanh_fp16<vec_length,unroll>::run(pout,pin);
    }
}

template void tanh_tpp<32>::ref(void *, void *, int64_t);
template void tanh_tpp<32>::ref_(void *, void *, int64_t);
}