import os
import sys
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")

def transpose_tile_weight(weight):
    row = weight.shape[0]
    col = weight.shape[1]

    col_step = int(col / 64)
    col_tile = int(row / 64)

    weight = weight.view(col_tile * 16, 4, col)
    weight = weight.transpose(1, 2).reshape(col_tile * 16, col * 4)
    weight = weight.view(col_tile, 16, col * 4)
    weight = weight.view(col_tile, 16, col_step, 4, 64)
    weight = weight.permute(2, 3, 0, 1, 4).contiguous()

    return weight

def test_amx_linear():
    batch_size = 128
    hidden_size = 4096

    for input_size in [1024, 2048]:
        x = torch.tensor(np.arange(input_size * batch_size).reshape(batch_size, input_size), dtype=torch.int8)
        w = torch.tensor(np.arange(input_size * hidden_size).reshape(hidden_size, input_size), dtype=torch.int8)
        b = torch.tensor(np.arange(hidden_size) / hidden_size, dtype=torch.float32)
        y_expected = torch.ops.intel_mlperf.linear(x, torch.ops.intel_mlperf.prepack_linear_weight(w), b, 0.1, None)
        # y = torch.ops.intel_mlperf.amx_linear(x.reshape(batch_size, 1, input_size), transpose_tile_weight(w.transpose(1, 0)), b, 0.1, False, 1.0)
        y = torch.ops.intel_mlperf.amx_linear_i8o32(x, transpose_tile_weight(w.transpose(1, 0)), b, 0.1)
        # np.testing.assert_equal(y.reshape(-1, hidden_size).numpy(), y_expected.numpy())
        np.testing.assert_equal(y.numpy(), y_expected.numpy())

# cannot pytest directly due to core nums requirement.
# numactl -C 0-3 python test/pyfiles/test_amx_linear.py
test_amx_linear()
