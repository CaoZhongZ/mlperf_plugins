# numactl -C 0-3 pytest test/pyfiles/test_amx_linear.py
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pytest

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(precision=10)


def transpose_tile_weight(weight, padding: bool = False):
    row = weight.shape[0]
    col = weight.shape[1]

    col_step = (col + 63) // 64
    col_tile = (row + 63) // 64
    if padding:
        pad_size = (0, col_step * 64 - col, 0, col_tile * 64 - row)
        weight = F.pad(weight, pad_size, "constant", 0.0)

    weight = weight.view(col_tile * 16, 4, col_step * 64)
    weight = weight.transpose(1, 2).reshape(col_tile * 16, col_step * 256)
    weight = weight.view(col_tile, 16, col_step * 256)
    weight = weight.view(col_tile, 16, col_step, 4, 64)
    weight = weight.permute(2, 3, 0, 1, 4).contiguous()

    return weight


def transpose_tile_weight_bf16(weight, padding: bool = False):
    row = weight.shape[0]
    col = weight.shape[1]

    col_step = (col + 31) // 32
    col_tile = (row + 31) // 32
    if padding:
        pad_size = (0, col_step * 32 - col, 0, col_tile * 32 - row)
        weight = F.pad(weight, pad_size, "constant", 0.0)

    weight = weight.view(col_tile * 16, 2, col_step * 32)
    weight = weight.transpose(1, 2).reshape(col_tile * 16, col_step * 64)
    weight = weight.view(col_tile, 16, col_step * 64)
    weight = weight.view(col_tile, 16, col_step, 2, 32)
    weight = weight.permute(2, 3, 0, 1, 4).contiguous()

    return weight


def get_weight(input_size, output_size, use_amx_linear=True, dtype=torch.int8):
    if use_amx_linear:
        if dtype == torch.int8:
            return transpose_tile_weight(
                torch.tensor(
                    np.arange(input_size * output_size).reshape(
                        input_size, output_size
                    ),
                    dtype=dtype,
                ).transpose(1, 0)
            )
        else:
            return transpose_tile_weight_bf16(
                torch.tensor(
                    np.arange(input_size * output_size).reshape(input_size, output_size)
                    / output_size
                    / input_size,
                    dtype=dtype,
                ).transpose(1, 0)
            )
    else:
        if dtype == torch.int8:
            return torch.tensor(
                np.arange(input_size * output_size).reshape(input_size, output_size),
                dtype=dtype,
            )
        else:
            return torch.tensor(
                np.arange(input_size * output_size).reshape(input_size, output_size)
                / output_size
                / input_size,
                dtype=dtype,
            )


def get_2d_input(input_size, output_size, dtype=torch.int8):
    if dtype == torch.int8:
        return torch.tensor(
            np.arange(input_size * output_size).reshape(input_size, output_size),
            dtype=dtype,
        )
    else:
        return torch.tensor(
            np.arange(input_size * output_size).reshape(input_size, output_size)
            / output_size
            / input_size,
            dtype=dtype,
        )


@pytest.mark.parametrize("batch_size", [32, 30, 33, 64])
def test_amx_linear(batch_size):
    input_size = 2048
    output_size = 4096

    x = torch.tensor(
        np.ones(input_size * batch_size).reshape(batch_size, input_size),
        dtype=torch.int8,
    )
    w = torch.tensor(
        np.arange(input_size * output_size).reshape(output_size, input_size)
        / input_size,
        dtype=torch.int8,
    )
    b = torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
    y_expected = torch.ops.intel_mlperf.linear(
        x, torch.ops.intel_mlperf.prepack_linear_weight(w), b, 1e-5, None
    )
    # y = torch.ops.intel_mlperf.amx_linear(x.reshape(batch_size, 1, input_size), transpose_tile_weight(w.transpose(1, 0)), b, 1e-5, False, 1.0)
    y = torch.ops.intel_mlperf.amx_linear_i8o32(
        x, transpose_tile_weight(w.transpose(1, 0)), b, 1e-5
    )
    # np.testing.assert_equal(y.reshape(-1, output_size).numpy(), y_expected.numpy())
    np.testing.assert_equal(y.numpy(), y_expected.numpy())


@pytest.mark.parametrize("batch_size,output_size", [(32, 1024), (128, 61)])
def test_amx_linear_padding(batch_size, output_size):
    input_size = 240

    x = torch.tensor(
        np.ones(input_size * batch_size).reshape(batch_size, input_size),
        dtype=torch.int8,
    )
    w = torch.tensor(
        np.arange(input_size * output_size).reshape(output_size, input_size),
        dtype=torch.int8,
    )
    b = torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
    b_pad = F.pad(b, (0, 64 - output_size % 64), "constant", 0)
    y_expected = torch.ops.intel_mlperf.linear(
        x, torch.ops.intel_mlperf.prepack_linear_weight(w), b, 0.1, None
    )
    pad_size = (0, 64 - x.shape[-1] % 64)
    x = F.pad(x, pad_size, "constant", 0)
    y = torch.ops.intel_mlperf.amx_linear_i8o32(
        x, transpose_tile_weight(w.transpose(1, 0), True), b_pad, 0.1
    )
    y_expected = F.pad(y_expected, (0, y.shape[-1] - output_size), "constant", 0)
    np.testing.assert_equal(y.numpy(), y_expected.numpy())


@pytest.mark.parametrize("batch_size,output_size", [(32, 1024), (128, 29)])
def test_amx_linear_bf16_padding(batch_size, output_size):
    input_size = 121

    x = torch.tensor(
        np.ones(input_size * batch_size).reshape(batch_size, input_size),
        dtype=torch.bfloat16,
    )
    w = torch.tensor(
        np.arange(input_size * output_size).reshape(output_size, input_size)
        / input_size,
        dtype=torch.bfloat16,
    )
    b = torch.tensor(np.arange(output_size), dtype=torch.float32)
    b_pad = F.pad(b, (0, 32 - output_size % 32), "constant", 0)
    y_expected = torch.matmul(x.float(), w.transpose(1, 0).float()) + b
    pad_size = (0, 32 - x.shape[-1] % 32)
    x = F.pad(x, pad_size, "constant", 0)
    y = torch.ops.intel_mlperf.amx_linear_i16o32(
        x, transpose_tile_weight_bf16(w.transpose(1, 0), True), b_pad
    )
    y_expected = F.pad(y_expected, (0, y.shape[-1] - output_size), "constant", 0)
    np.testing.assert_equal(y.float().numpy(), y_expected.float().numpy())


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_amx_linear_bf16_relu(batch_size):
    input_size = 2048
    hidden_size = 1024
    output_size = 4 * hidden_size

    x = torch.tensor(
        np.ones(input_size * batch_size).reshape(batch_size, input_size),
        dtype=torch.bfloat16,
    )
    w = torch.tensor(
        np.arange(input_size * output_size).reshape(output_size, input_size)
        / input_size,
        dtype=torch.bfloat16,
    )
    b = torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
    y_expected = torch.relu(
        torch.ops.intel_mlperf.linear(
            x, torch.ops.intel_mlperf.prepack_linear_weight(w), b, None, None
        )
        + torch.ops.intel_mlperf.linear(
            x, torch.ops.intel_mlperf.prepack_linear_weight(w), b, None, None
        )
    )
    y = torch.ops.intel_mlperf.amx_linear_bf16_accum_relu(
        x,
        transpose_tile_weight_bf16(w.transpose(1, 0)),
        x,
        transpose_tile_weight_bf16(w.transpose(1, 0)),
        b + b,
    )
    np.testing.assert_equal(y.float().numpy(), y_expected.float().numpy())


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_lstm_amx_int8(batch_size):
    dtype = torch.int8
    input_size = 1024
    hidden_size = 1024
    output_size = hidden_size * 4
    seq_len = 2
    layer = 3
    count = 1  # for cache test

    ww_i = [get_weight(output_size, input_size, dtype=dtype) for i in range(layer)]
    b_i = [
        torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    ww_h = [get_weight(output_size, hidden_size, dtype=dtype) for i in range(layer)]
    b_h = [
        torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    for count in range(count):
        # x = [get_2d_input(batch_size, input_size, dtype=dtype) for i in range(seq_len)]
        x = torch.tensor(
            np.ones(seq_len * input_size * batch_size).reshape(
                seq_len, batch_size, input_size
            ),
            dtype=dtype,
        )
        h = [get_2d_input(batch_size, hidden_size, dtype=dtype) for i in range(layer)]
        c = [get_2d_input(batch_size, hidden_size, torch.half) for i in range(layer)]
        xx = torch.tensor(
            np.ones(seq_len * input_size * batch_size).reshape(
                seq_len, batch_size, input_size
            ),
            dtype=dtype,
        )
        hh = [get_2d_input(batch_size, hidden_size, dtype=dtype) for i in range(layer)]
        cc = [get_2d_input(batch_size, hidden_size, torch.half) for i in range(layer)]
        for i in range(layer):
            last_layer = i == layer - 1
            y = []
            for j in range(seq_len):
                y.append(
                    torch.ops.intel_mlperf.amx_linear_i8o32(x[j], ww_i[i], b_i[i], 0.01)
                )
            for j in range(seq_len):
                temp = torch.ops.intel_mlperf.amx_linear_i8o32(
                    h[i], ww_h[i], b_h[i], 0.01
                )
                y[j] += temp

                it, ft, gt, ot = y[j].chunk(4, 1)
                it, x[j], h[i], c[i] = torch.ops.intel_mlperf.lstm_postop(
                    it, ft, gt, ot, c[i], 0.01, 0.01, last_layer
                )
                if last_layer:
                    x = x.to(torch.bfloat16)
                    x[j] = it
            xx, hh[i], cc[i] = torch.ops.intel_mlperf.lstm_layer_amx_int8(
                xx,
                hh[i],
                cc[i],
                ww_i[i],
                ww_h[i],
                b_i[i],
                b_h[i] + b_i[i],
                0.01,
                0.01,
                0.01,
                last_layer,
            )
            if last_layer:
                xx = xx.float()
                x = x.float()
            for j in range(seq_len):
                np.testing.assert_equal(
                    xx[j].numpy(),
                    x[j].numpy(),
                    err_msg=f"layer{i} seq{j}",
                )
            np.testing.assert_equal(hh[i].numpy(), h[i].numpy(), err_msg=f"layer{i}")
            np.testing.assert_equal(cc[i].numpy(), c[i].numpy(), err_msg=f"layer{i}")


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_lstm_amx_bf16(batch_size):
    dtype = torch.bfloat16
    input_size = 128
    hidden_size = 128
    output_size = hidden_size * 4
    seq_len = 1
    layer = 1
    count = 1  # for cache test

    # tmp_input = torch.load("./tmp_input.pt")
    # tmp_output = torch.load("./tmp_output.pt")
    # tmp_weights = torch.load("./tmp_weights.pt")
    ww_i = [get_weight(output_size, input_size, dtype=dtype) for i in range(layer)]
    # ww_i = [transpose_tile_weight_bf16(w[0].clone().transpose(1, 0).detach().requires_grad_(False).to(torch.bfloat16)) for w in tmp_weights]
    ww_h = [get_weight(output_size, hidden_size, dtype=dtype) for i in range(layer)]
    # ww_h = [transpose_tile_weight_bf16(w[1].clone().transpose(1, 0).detach().requires_grad_(False).to(torch.bfloat16)) for w in tmp_weights]
    w_i = []
    w_h = []
    for i in range(layer):
        (prepacked_w_i, prepacked_w_h,) = torch.ops.intel_mlperf.prepack_lstm_weights(
            get_weight(output_size, input_size, False, dtype=dtype),
            get_weight(output_size, hidden_size, False, dtype=dtype),
            # tmp_weights[i][0].clone().detach().requires_grad_(False).to(torch.bfloat16), tmp_weights[i][1].clone().detach().requires_grad_(False).to(torch.bfloat16)
        )
        # prepacked_w_i[...] = 0
        # ww_i[i][...] = 0
        # prepacked_w_h[...] = 0
        # ww_h[i][...] = 0
        w_i.append(prepacked_w_i)
        w_h.append(prepacked_w_h)
    b_i = [
        # tmp_weights[i][2].clone().detach().requires_grad_(False)
        torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    b_h = [
        # tmp_weights[i][3].clone().detach().requires_grad_(False)
        torch.tensor(np.arange(output_size) / output_size, dtype=torch.float32)
        for i in range(layer)
    ]
    for count in range(count):
        # x = [get_2d_input(batch_size, input_size, dtype=dtype) for i in range(seq_len)]
        x = torch.tensor(
            np.ones(seq_len * input_size * batch_size).reshape(
                seq_len, batch_size, input_size
            ),
            dtype=dtype,
        )
        # x = torch.clone(tmp_input[0]).detach().requires_grad_(False)
        h = [get_2d_input(batch_size, hidden_size, dtype=dtype) for i in range(layer)]
        # h = [torch.clone(tmp_input[1][i]).detach().requires_grad_(False) for i in range(layer)]
        c = [get_2d_input(batch_size, hidden_size, dtype=dtype) for i in range(layer)]
        # c = [torch.clone(tmp_input[2][i]).detach().requires_grad_(False) for i in range(layer)]
        xx = torch.tensor(
            np.ones(seq_len * input_size * batch_size).reshape(
                seq_len, batch_size, input_size
            ),
            dtype=dtype,
        )
        # xx = torch.clone(tmp_input[0]).detach().requires_grad_(False)
        hh = [get_2d_input(batch_size, hidden_size, dtype=dtype) for i in range(layer)]
        # hh = [torch.clone(tmp_input[1][i]).detach().requires_grad_(False) for i in range(layer)]
        cc = [
            get_2d_input(batch_size, hidden_size, dtype=torch.float32)
            for i in range(layer)
        ]
        # cc = [torch.clone(tmp_input[2][i]).detach().requires_grad_(False) for i in range(layer)]
        """
        y = [x[i] for i in range(seq_len)]
        x, h, c = torch.ops.intel_mlperf.lstm(
            x, h, c, [[w_i[i], w_h[i], b_i[i], b_h[i]] for i in range(layer)]
        )
        xx, hh, cc = torch.ops.intel_mlperf.lstm_amx_bf16(
            xx,
            hh,
            cc,
            [[ww_i[i], ww_h[i], b_i[i], b_h[i] + b_i[i]] for i in range(layer)],
        )
        print(x[0], xx[0])
        for j in range(seq_len):
            np.testing.assert_equal(
                x[j].float().numpy(),
                tmp_output[0][j].float().numpy(),
            # atol= 1 / 128,
                err_msg=f"seq{j}",
            )
        for i in range(layer):
            np.testing.assert_equal(
                hh[i].float().numpy(), h[i].float().numpy(), atol=1/128, err_msg=f"layer{i}"
            )
            np.testing.assert_equal(
                cc[i].float().numpy(), c[i].float().numpy(), atol=1/128, err_msg=f"layer{i}"
            )
        for j in range(seq_len):
            np.testing.assert_equal(
                xx[j].float().numpy(),
                x[j].float().numpy(),
                atol=1/128,
                err_msg=f"layer{i} seq{j}",
            )
        continue
        """
        for i in range(layer):
            x, h[i], c[i] = torch.ops.intel_mlperf.lstm_layer_1dnn(
                x,
                h[i],
                c[i],
                w_i[i],
                w_h[i],
                b_i[i],
                b_h[i],
            )
            """
            y = []
            for j in range(seq_len):
                y.append(
                    torch.ops.intel_mlperf.amx_linear_bf16(x[j], ww_i[i], b_i[i])
                )
            for j in range(seq_len):
                temp = torch.ops.intel_mlperf.amx_linear_bf16(
                    h[i], ww_h[i], b_h[i]
                )
                y[j] += temp

                it, ft, gt, ot = y[j].chunk(4, 1)
                it = torch.sigmoid(it)
                ft = torch.sigmoid(ft)
                gt = torch.tanh(gt)
                ot = torch.sigmoid(ot)
                ct = (ft * c[i].to(torch.float32)) + (it * gt)
                c[i] = ct.to(torch.bfloat16)
                h[i] = (ot * torch.tanh(ct)).to(torch.bfloat16)
                x[j] = h[i]
                # it, x[j], h[i], c[i] = torch.ops.intel_mlperf.lstm_postop(
                    # it, ft, gt, ot, c[i], 0.01, 0.01, last_layer
                # )
            """
            xx, hh[i], cc[i] = torch.ops.intel_mlperf.lstm_layer_amx_bf16(
                xx,
                hh[i],
                cc[i],
                ww_i[i],
                ww_h[i],
                b_i[i],
                b_h[i] + b_i[i],
            )

            for j in range(seq_len):
                np.testing.assert_allclose(
                    xx[j].float().numpy(),
                    x[j].float().numpy(),
                    atol=1 / 128,
                    err_msg=f"layer{i} seq{j}",
                )
            np.testing.assert_allclose(
                hh[i].float().numpy(),
                xx[-1].float().numpy(),
                atol=1 / 128,
                err_msg=f"layer{i}",
            )
            np.testing.assert_allclose(
                hh[i].float().numpy(),
                h[i].float().numpy(),
                atol=1 / 128,
                err_msg=f"layer{i}",
            )

            np.testing.assert_allclose(
                cc[i].float().numpy(),
                c[i].float().numpy(),
                atol=1 / 128,
                err_msg=f"layer{i}",
            )
