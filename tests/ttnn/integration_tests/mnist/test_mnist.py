# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.mnist.reference.mnist import MnistModel
from models.experimental.functional_mnist.tt import ttnn_functional_mnist
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [2],
)
def test_mnist(reset_seeds, batch_size, model_location_generator, mesh_device):
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # x = torch.randn((1, 1, 28, 28), dtype=torch.bfloat16).float()  # random_inputs
    x, labels = next(iter(dataloader))

    torch_output = model(x)

    parameters = ttnn_functional_mnist.custom_preprocessor(state_dict, weights_mesh_mapper, mesh_device)
    # print("@@@@@@@@@@@@@@@@@@",parameters)

    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper, device=mesh_device)

    tt_output = ttnn_functional_mnist.mnist(
        mesh_device, batch_size, x, parameters, mesh_mapper=inputs_mesh_mapper, mesh_composer=output_mesh_composer
    )

    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer).permute(1, 2, 0, 3).squeeze(0).squeeze(0)

    assert_with_pcc(torch_output, tt_output, 0.99)
