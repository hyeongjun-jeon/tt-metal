# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_example(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    # test output1
    output1, _ = ttnn.prim.example(input_tensor)
    output1 = ttnn.to_torch(output1)

    assert_with_pcc(torch_output_tensor, output1, 0.99)

    # test output2
    # _, output2 = ttnn.prim.example(input_tensor)
    # output2 = ttnn.to_torch(output2)

    # assert_with_pcc(torch_output_tensor, output2, 0.99)
