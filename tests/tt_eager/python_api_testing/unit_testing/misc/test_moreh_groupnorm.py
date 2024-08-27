# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.utility_functions import comp_allclose
from loguru import logger


from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import to_cpu, to_npu


def tt_groupnorm(input, num_groups, eps=1e-05, device=None):
    npu_input = to_npu(input, device)

    # Forward
    npu_output, _, _ = ttnn.experimental.operations.primary.moreh_groupnorm(
        npu_input, num_groups, eps, are_required_outputs=(True, False, False)
    )

    tt_output = to_cpu(npu_output, input.shape)

    return tt_output


def run_moreh_group(N, C_num_groups, H, W, eps, device):
    C, num_groups = C_num_groups
    input_shape = (N, C, H, W)
    # cpu_input = torch.arange(N * C * H * W, dtype=torch.bfloat16).reshape(input_shape)
    cpu_input = torch.zeros(input_shape, dtype=torch.bfloat16)

    # actual
    actual_output = tt_groupnorm(
        cpu_input,
        num_groups,
        eps,
        device,
    )


@pytest.mark.parametrize(
    "C_num_groups",
    [
        [1, 1],
    ],
)
def test_moreh_groupnorm(C_num_groups, device):
    torch.manual_seed(2024)

    N = H = W = 1
    eps = 0.0
    run_moreh_group(N, C_num_groups, H, W, eps, device)
