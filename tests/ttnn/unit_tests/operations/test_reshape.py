import ttnn
import torch
import pytest
import math
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "act_shape",
    (
        ## mnist shapes
        [1, 1, 28, 28],
    ),
)
def test(reset_seeds, device, act_shape):
    x = torch.randn(act_shape, dtype=torch.bfloat16)
    weights = torch.randn((784, 120), dtype=torch.bfloat16)
    bias = torch.randn((120), dtype=torch.bfloat16)
    weights = ttnn.from_torch(weights, device=device, layout=ttnn.TILE_LAYOUT)
    bias = ttnn.from_torch(bias, device=device, layout=ttnn.TILE_LAYOUT)

    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16)

    tt_x1 = ttnn.reshape(tt, (tt.shape[0], 1, 1, 784))
    tt_x1 = ttnn.to_device(tt_x1, device=device)
    tt_x1 = ttnn.to_layout(tt_x1, layout=ttnn.TILE_LAYOUT)

    tt_reshape = ttnn.linear(
        tt_x1,
        weights,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt = ttnn.to_device(tt, device=device)
    tt_x2 = ttnn.reshape(tt, (tt.shape[0], 1, 1, 784))
    tt_x2 = ttnn.to_device(tt_x2, device=device)
    tt_x2 = ttnn.to_layout(tt_x2, layout=ttnn.TILE_LAYOUT)

    tt_reshape_device = ttnn.linear(
        tt_x2,
        weights,
        bias=bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_output = ttnn.to_torch(tt_reshape)
    tt_output_device = ttnn.to_torch(tt_reshape_device)

    assert_with_pcc(tt_output, tt_output_device, 1)
