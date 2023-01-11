import os
import sys
import pytest
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")


@pytest.mark.parametrize("seq_len", [16, 17, 25])
def test_frame_splicing(seq_len):
    batch_size = 3
    x = torch.tensor(
        np.arange(batch_size * seq_len * 2).reshape(batch_size, seq_len, 2),
        dtype=torch.float32,
    )
    y_expected = (x * x).sum(-1)
    y = torch.ops.intel_mlperf.power_spectrum(x)
    np.testing.assert_equal(y.numpy(), y_expected.numpy())
