# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def mnist(device, batch_size, x, parameters):
    # x = torch.reshape(x, [x.shape[0], 1, 1, 784])
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16)
    x = ttnn.reshape(x, (x.shape[0], 1, 1, 784))
    weights_tensor = parameters.fc1.weight
    weights_tensor = ttnn.to_device(weights_tensor, device=device)
    bias_tensor = parameters.fc1.bias
    bias_tensor = ttnn.to_device(bias_tensor, device=device)
    x = ttnn.to_device(x, device=device)
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.linear(
        x,
        weights_tensor,
        bias=bias_tensor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.relu(x)

    weights_tensor = parameters.fc2.weight
    weights_tensor = ttnn.to_device(weights_tensor, device=device)

    bias_tensor = parameters.fc2.bias
    bias_tensor = ttnn.to_device(bias_tensor, device=device)

    x = ttnn.linear(
        x,
        weights_tensor,
        bias=bias_tensor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.relu(x)

    weights_tensor = parameters.fc3.weight
    weights_tensor = ttnn.to_device(weights_tensor, device=device)

    bias_tensor = parameters.fc3.bias
    bias_tensor = ttnn.to_device(bias_tensor, device=device)

    x = ttnn.linear(
        x,
        weights_tensor,
        bias=bias_tensor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.relu(x)

    x = ttnn.softmax(x)

    return x
