# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def preprocess_weight(weight, mesh_mapper, mesh_device):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(
        weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return weight


def preprocess_bias(parameter, mesh_mapper, mesh_device):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(
        parameter, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return parameter


def custom_preprocessor(state_dict, mesh_mapper, mesh_device):
    parameters = {}
    for name, parameter in state_dict.items():
        if "weight" in name:
            parameters[name] = preprocess_weight(parameter, mesh_mapper, mesh_device)
        elif "bias" in name:
            parameters[name] = preprocess_bias(parameter, mesh_mapper, mesh_device)

    return parameters


def mnist(mesh_device, batch_size, x, parameters, mesh_mapper=None, mesh_composer=None):
    # x = torch.reshape(x, [x.shape[0], 1, 1, 784])
    x = ttnn.from_device(x)
    x = ttnn.reshape(x, (x.shape[0], 1, 1, 784))
    weights_tensor = parameters[f"fc1.weight"]
    # weights_tensor = ttnn.to_device(weights_tensor, device=mesh_device)
    bias_tensor = parameters[f"fc1.bias"]
    # bias_tensor = ttnn.to_device(bias_tensor, device=mesh_device)
    x = ttnn.to_device(x, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.linear(
        x,
        weights_tensor,
        bias=bias_tensor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.relu(x)

    weights_tensor = parameters[f"fc2.weight"]
    # weights_tensor = ttnn.to_device(weights_tensor, device=device)

    bias_tensor = parameters[f"fc2.bias"]
    # bias_tensor = ttnn.to_device(bias_tensor, device=device)

    x = ttnn.linear(
        x,
        weights_tensor,
        bias=bias_tensor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.relu(x)

    weights_tensor = parameters[f"fc3.weight"]
    # weights_tensor = ttnn.to_device(weights_tensor, device=device)

    bias_tensor = parameters[f"fc3.bias"]
    # bias_tensor = ttnn.to_device(bias_tensor, device=device)

    x = ttnn.linear(
        x,
        weights_tensor,
        bias=bias_tensor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.relu(x)

    x = ttnn.softmax(x)

    return x
