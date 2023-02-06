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
    length = torch.tensor(
        [seq_len, seq_len - hop_length, seq_len - hop_length - 1], dtype=torch.int32
    )
    for i in range(batch_size):
        x[i, length[i] :] = 0
    y_expected = torch.cat(
        (x[:, 0].unsqueeze(1), x[:, 1:] - preemph * x[:, :-1]), dim=1
    )
    padded_y_expected = torch.zeros((batch_size, seq_len + n_fft))
    for i in range(batch_size):
        tmp = torch.nn.functional.pad(
            y_expected[i : i + 1, : length[i]], (n_fft // 2, n_fft // 2), "reflect"
        )
        padded_y_expected[i : i + 1, : tmp.shape[1]] = tmp
    y = torch.ops.intel_mlperf.preemphasis(
        x, length, coeff=preemph, pad_size=n_fft // 2
    )
    np.testing.assert_allclose(y.numpy(), padded_y_expected.numpy(), rtol=1e-6)
