import os
import sys
import pytest
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")


# TODO: fix seq_len of 49 and 50
@pytest.mark.parametrize("seq_len", [48])
def test_frame_splicing(seq_len):
    frame_splicing = 3
    x = torch.tensor(np.arange(seq_len * 6).reshape(3, 2, seq_len), dtype=torch.float32)
    seq = [x]
    for n in range(1, frame_splicing):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    y_expected = torch.cat(seq, dim=1)[:, :, ::frame_splicing].contiguous()
    y = torch.ops.intel_mlperf.frame_splicing(x, frame_splicing)
    np.testing.assert_equal(y.numpy(), y_expected.numpy())
