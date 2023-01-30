import os
import sys
import pytest
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(precision=10)


@pytest.mark.parametrize("seq_len", [17, 16, 18])
def test_preemphasis(seq_len):
    batch_size = 3
    preemph = 0.97
    n_fft = 6
    hop_length = 4
    win_length = 2
    x = torch.tensor(
        np.arange(batch_size * seq_len).reshape(batch_size, seq_len),
        dtype=torch.float32,
    )
    y_expected = torch.cat(
        (x[:, 0].unsqueeze(1), x[:, 1:] - preemph * x[:, :-1]), dim=1
    )
    y_expected = torch.nn.functional.pad(
        y_expected.view(1, batch_size, seq_len), (n_fft // 2, n_fft // 2), "reflect"
    ).view(batch_size, seq_len + n_fft)
    y = torch.ops.intel_mlperf.preemphasis(x, coeff=preemph, pad_size=n_fft // 2)
    np.testing.assert_allclose(y.numpy(), y_expected.numpy(), rtol=1e-6)


@pytest.mark.parametrize("seq_len", [96, 3200 + 256])
def test_stft(seq_len):
    batch_size = 3
    n_fft = 32
    hop_length = 16
    win_length = 16
    x = torch.tensor(
        np.ones(seq_len * batch_size).reshape(batch_size, seq_len),
        dtype=torch.float32,
    )
    window = torch.tensor(
        np.arange(win_length),
        dtype=torch.float32,
    )
    pad_size = (n_fft - win_length) // 2
    padded_window = torch.nn.functional.pad(window, (pad_size, pad_size), "constant", 0)

    n_frames = 1 + (seq_len - n_fft) // hop_length
    # y_expected = x.as_strided((batch_size, n_frames, n_fft), (x.stride(0), hop_length, 1)) * padded_window
    y_expected = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        center=False,
        win_length=win_length,
        window=window,
    )
    y = torch.ops.intel_mlperf.stft(x, padded_window, n_fft, hop_length, win_length)
    # for i in range(batch_size):
        # print(y[i, :, :n_fft-hop_length], y_expected[i, :, :n_fft - hop_length])
        # np.testing.assert_equal(y[i, :, :n_fft-hop_length].numpy(), y_expected[i, :, :n_fft - hop_length].numpy())
    np.testing.assert_equal(y.numpy(), y_expected.numpy())


@pytest.mark.parametrize("seq_len", [48, 49, 50, 52])
def test_frame_splicing(seq_len):
    batch_size = 3
    freq = 2
    frame_splicing = 3
    x = torch.tensor(
        np.arange(seq_len * batch_size * freq).reshape(batch_size, freq, seq_len),
        dtype=torch.float32,
    )
    seq = [x]
    for n in range(1, frame_splicing):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    y_expected = torch.cat(seq, dim=1)[:, :, ::frame_splicing].contiguous()
    y = torch.ops.intel_mlperf.frame_splicing(x, frame_splicing)
    np.testing.assert_equal(y.numpy(), y_expected.numpy())


@pytest.mark.parametrize("seq_len", [16, 17, 25])
def test_power_spectrum(seq_len):
    batch_size = 3
    x = torch.tensor(
        np.arange(batch_size * seq_len * 2).reshape(batch_size, seq_len, 2),
        dtype=torch.float32,
    )
    y_expected = (x * x).sum(-1)
    y = torch.ops.intel_mlperf.power_spectrum(x)
    np.testing.assert_equal(y.numpy(), y_expected.numpy())
