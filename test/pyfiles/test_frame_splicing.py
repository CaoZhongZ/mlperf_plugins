import os
import sys
import pytest
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(precision=10)


@pytest.mark.parametrize("seq_len", [48, 49, 50, 51, 52, 97])
def test_frame_splicing(seq_len):
    batch_size = 3
    feat = 2
    frame_splicing = 3
    x = torch.tensor(
        np.arange(seq_len * batch_size * feat).reshape(batch_size, feat, seq_len),
        dtype=torch.float32,
    )
    length = torch.tensor([seq_len, seq_len - 1, seq_len - 2], dtype=torch.int32)
    if seq_len > 96:
        length[0] -= 48
    y = torch.ops.intel_mlperf.frame_splicing(x, length, frame_splicing)
    for i in range(batch_size):
        x[i, :, length[i] :] = 0
    seq = [x]
    for n in range(1, frame_splicing):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    y_expected = torch.cat(seq, dim=1)[:, :, ::frame_splicing].contiguous()
    np.testing.assert_equal(y.numpy(), y_expected.numpy())
