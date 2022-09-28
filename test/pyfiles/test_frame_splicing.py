import os
import sys
import numpy as np
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
torch.ops.load_library(script_dir + "/../../build/libmlperf_plugins.so")


def test_frame_splicing():
    frame_splicing = 3
    for seq_len in [48, 49, 50]:
        x = torch.tensor(np.arange(seq_len * 6).reshape(3, 2, seq_len), dtype=torch.float32)
        seq = [x]
        for n in range(1, frame_splicing):
            tmp = torch.zeros_like(x)
            tmp[:, :, :-n] = x[:, :, n:]
            seq.append(tmp)
        y_expected = torch.cat(seq, dim=1)[:, :, ::frame_splicing].contiguous()
        y = torch.ops.intel_mlperf.frame_splicing(x, frame_splicing)
        np.testing.assert_equal(y.numpy(), y_expected.numpy())
