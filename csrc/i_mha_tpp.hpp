#pragma once

#include <string.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>

namespace intel_mlperf {

enum class status_t { success, failed };

status_t amx_per_head(const void *qkv_ptr, int ldqkv, void *a_ptr, size_t sl,
                      float M, float oscale, int32_t att_mask, float M2);

}