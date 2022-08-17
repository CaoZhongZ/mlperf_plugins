#pragma once

#include <cstdlib>

namespace intel_mlperf {

struct task_load {
  size_t start_;
  size_t load_;
};

static inline task_load balance(std::div_t chunk, int core_id, int n_cores) {
  int step = core_id < chunk.rem;
  auto load = chunk.quo + step;
  auto start = load * core_id + (1-step) * chunk.rem;

  return {start, load};
}

}
