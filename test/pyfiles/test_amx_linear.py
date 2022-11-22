# numactl -C 0-3 pytest test/pyfiles/test_amx_linear.py
import os
import sys
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")


def transpose_tile_weight(weight, padding=False):
    row = weight.shape[0]
    col = weight.shape[1]

    col_step = (col + 63) // 64
    col_tile = (row + 63) // 64
    if padding:
        pad_size = (0, col_step * 64 - col, 0, col_tile * 64 - row)
        weight = torch.nn.functional.pad(weight, pad_size, "constant", 0)
        # print("prepack weight padding", weight.shape, row, col)

    weight = weight.view(col_tile * 16, 4, col)
    weight = weight.transpose(1, 2).reshape(col_tile * 16, col * 4)
    weight = weight.view(col_tile, 16, col * 4)
    weight = weight.view(col_tile, 16, col_step, 4, 64)
    weight = weight.permute(2, 3, 0, 1, 4).contiguous()
    # print("after prepack weight shape", weight.shape, row, col)

    return weight


def get_weight(input_size, output_size, use_amx_linear=True):
    if use_amx_linear:
        return transpose_tile_weight(
            torch.tensor(
                np.ones(input_size * output_size).reshape(output_size, input_size),
                dtype=torch.int8,
            ).transpose(1, 0)
        )
    else:
        return torch.ops.intel_mlperf.prepack_linear_weight(
            torch.tensor(
                np.ones(input_size * output_size).reshape(output_size, input_size),
                dtype=torch.int8,
            )
        )


def test_amx_linear():
    batch_size = 32
    input_size = 2048
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


def test_amx_linear_padding():
    batch_size = 128
    input_size = 240
    hidden_size = 1024
    output_size = 4 * hidden_size

    x = torch.tensor(
        np.ones(input_size * batch_size).reshape(batch_size, input_size),
        dtype=torch.int8,
    )
    w = torch.tensor(
        np.ones(input_size * output_size).reshape(output_size, input_size),
        dtype=torch.int8,
    )
    b = torch.tensor(np.ones(output_size) / output_size, dtype=torch.float32)
    y_expected = torch.ops.intel_mlperf.linear(
        x, torch.ops.intel_mlperf.prepack_linear_weight(w), b, 0.1, None
    )
    # y = torch.ops.intel_mlperf.amx_linear(x.reshape(batch_size, 1, input_size), transpose_tile_weight(w.transpose(1, 0)), b, 0.1, False, 1.0)
    pad_size = (0, 64 - x.shape[-1] % 64)
    x = torch.nn.functional.pad(x, pad_size, "constant", 0)
    # print("first layer padding x.shape", x.shape)
    y = torch.ops.intel_mlperf.amx_linear_i8o32(
        x, transpose_tile_weight(w.transpose(1, 0), True), b, 0.1
    )
    # np.testing.assert_equal(y.reshape(-1, output_size).numpy(), y_expected.numpy())
    np.testing.assert_equal(y.numpy(), y_expected.numpy())


def test_lstm_int8():
    batch_size = 32
    input_size = 2048
    output_size = 4096
    hidden_size = 1024
    seq_len = 2
    layer = 3

    w_i = [get_weight(input_size, output_size) for i in range(layer)]
    b_i = [
        torch.tensor(np.ones(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    w_h = [get_weight(hidden_size, output_size) for i in range(layer)]
    b_h = [
        torch.tensor(np.ones(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    for count in range(1000):
        x = [
            torch.tensor(
                np.ones(input_size * batch_size).reshape(batch_size, input_size),
                dtype=torch.int8,
            )
            for i in range(seq_len)
        ]
        h = [
            torch.tensor(
                np.ones(hidden_size * batch_size).reshape(batch_size, hidden_size),
                dtype=torch.int8,
            )
            for i in range(layer)
        ]
        c = [
            torch.tensor(
                np.ones(hidden_size * batch_size).reshape(batch_size, hidden_size),
                dtype=torch.half,
            )
            for i in range(layer)
        ]
        xx = torch.tensor(
            np.ones(seq_len * input_size * batch_size).reshape(
                seq_len, batch_size, input_size
            ),
            dtype=torch.int8,
        )
        hh = [
            torch.tensor(
                np.ones(hidden_size * batch_size).reshape(batch_size, hidden_size),
                dtype=torch.int8,
            )
            for i in range(layer)
        ]
        cc = [
            torch.tensor(
                np.ones(hidden_size * batch_size).reshape(batch_size, hidden_size),
                dtype=torch.half,
            )
            for i in range(layer)
        ]
        # y = [x[i] for i in range(seq_len)]
        for i in range(layer):
            last_layer = i == layer - 1
            y = []
            for j in range(seq_len):
                y.append(
                    torch.ops.intel_mlperf.amx_linear_i8o32(x[j], w_i[i], b_i[i], 0.1)
                )
            for j in range(seq_len):
                temp = torch.ops.intel_mlperf.amx_linear_i8o32(
                    h[i], w_h[i], b_h[i], 0.1
                )
                y[j] += temp

                it, ft, gt, ot = y[j].chunk(4, 1)
                it, x[j], h[i], c[i] = torch.ops.intel_mlperf.lstm_postop(
                    it, ft, gt, ot, c[i], 0.1, 0.1, last_layer
                )
                if last_layer:
                    x[j] = it
            xx, hh[i], cc[i] = torch.ops.intel_mlperf.lstm_layer_int8(
                xx,
                hh[i],
                cc[i],
                w_i[i],
                w_h[i],
                b_i[i],
                b_h[i],
                0.1,
                0.1,
                0.1,
                last_layer,
            )
            np.testing.assert_equal(hh[i].numpy(), h[i].numpy())
            np.testing.assert_equal(cc[i].numpy(), c[i].numpy())
            for j in range(seq_len):
                np.testing.assert_equal(xx[j].numpy(), x[j].numpy())
