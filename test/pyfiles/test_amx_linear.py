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


def get_weight(input_size, output_size, use_amx_linear=True):
    if use_amx_linear:
        return transpose_tile_weight(
            torch.tensor(
                np.arange(input_size * output_size).reshape(output_size, input_size),
                dtype=torch.int8,
            ).transpose(1, 0)
        )
    else:
        return torch.ops.intel_mlperf.prepack_linear_weight(
            torch.tensor(
                np.arange(input_size * output_size).reshape(output_size, input_size),
                dtype=torch.int8,
            )
        )


def test_amx_linear():
    batch_size = 128
    input_size = 1024
    hidden_size = 1024
    output_size = 4 * hidden_size

    x = torch.tensor(
        np.arange(input_size * batch_size).reshape(batch_size, input_size),
        dtype=torch.int8,
    )
    w = torch.tensor(
        np.arange(input_size * output_size).reshape(output_size, input_size),
        dtype=torch.int8,
    )
    b = torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
    y_expected = torch.ops.intel_mlperf.linear(
        x, torch.ops.intel_mlperf.prepack_linear_weight(w), b, 0.1, None
    )
    # y = torch.ops.intel_mlperf.amx_linear(x.reshape(batch_size, 1, input_size), transpose_tile_weight(w.transpose(1, 0)), b, 0.1, False, 1.0)
    y = torch.ops.intel_mlperf.amx_linear_i8o32(
        x, transpose_tile_weight(w.transpose(1, 0)), b, 0.1
    )
    # np.testing.assert_equal(y.reshape(-1, output_size).numpy(), y_expected.numpy())
    np.testing.assert_equal(y.numpy(), y_expected.numpy())


def test_lstm_int8():
    batch_size = 32
    input_size = 2048
    output_size = 4096
    hidden_size = 1024
    seq_len = 2
    layer = 1

    w_i = [get_weight(input_size, output_size) for i in range(layer)]
    b_i = [
        torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    w_h = [get_weight(hidden_size, output_size) for i in range(layer)]
    b_h = [
        torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    for count in range(1000):
        x = [
            torch.tensor(
                np.arange(input_size * batch_size).reshape(batch_size, input_size),
                dtype=torch.int8,
            )
            for i in range(seq_len)
        ]
        h = torch.tensor(
            np.arange(layer * hidden_size * batch_size).reshape(
                layer, batch_size, hidden_size
            ),
            dtype=torch.int8,
        )
        c = torch.tensor(
            np.arange(layer * batch_size * output_size).reshape(
                layer, batch_size, output_size
            ),
            dtype=torch.half,
        )
        # y_expected = torch.ops.intel_mlperf.lstm_int8(
        # x,
        # h,
        # c,
        # [[w_i[i], w_h[i], b_i[i], b_h[i]] for i in range(layer)],
        # [0.1],
        # [0.1],
        # [0.1],
        # False,
        # False,
        # )
        y = [x[i] for i in range(seq_len)]
        for i in range(layer):
            for j in range(seq_len):
                # y[j] = torch.ops.intel_mlperf.linear(y[j], w_i[i], b_i[i], 0.1, None)
                y[j] = torch.ops.intel_mlperf.amx_linear_i8o32(
                    y[j], w_i[i], b_i[i], 0.1
                )
            for j in range(seq_len):
                # y[j] += torch.ops.intel_mlperf.linear(h[i], w_h[i], b_h[i], 0.1, None)
                y[j] += torch.ops.intel_mlperf.amx_linear_i8o32(
                    h[i], w_h[i], b_h[i], 0.1
                )

                it, ft, gt, ot = y[j].chunk(4, 1)
                y[j] = torch.ops.intel_mlperf.lstm_postop(
                    it, ft, gt, ot, c[i], 0.1, 0.1, False
                )


# cannot pytest directly due to core nums requirement.
# numactl -C 0-3 python test/pyfiles/test_amx_linear.py
# test_amx_linear()
test_lstm_int8()
