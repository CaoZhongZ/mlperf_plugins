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
