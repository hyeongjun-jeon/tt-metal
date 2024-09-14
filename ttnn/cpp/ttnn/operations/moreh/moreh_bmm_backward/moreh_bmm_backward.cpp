// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward.hpp"

#include "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_bmm_backward {
std::vector<std::optional<Tensor>> MorehBmmBackward::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mat2,
    const std::vector<bool>& are_required_outputs,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& mat2_grad,
    const std::optional<MemoryConfig>& input_grad_mem_config,
    const std::optional<MemoryConfig>& mat2_grad_mem_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    std::vector<std::optional<Tensor>> outputs(2);
    outputs.reserve(2);

    const bool input_requires_grad = are_required_outputs.at(0);
    const bool mat2_requires_grad = are_required_outputs.at(1);

    if (input_requires_grad) {
        TT_FATAL(input_grad.has_value(), "input_grad needs to have a value when input_requires_grad is True.");
        const auto& input_grad_tensor = input_grad.value();
        outputs[0] = ttnn::moreh_matmul(
            output_grad,
            mat2,
            false,
            true,
            input_grad_tensor,
            std::nullopt,
            input_grad_mem_config,
            compute_kernel_config);
    }

    if (mat2_requires_grad) {
        TT_FATAL(mat2_grad.has_value(), "mat2_grad needs to have a value when mat2_requires_grad is True.");
        const auto& mat2_grad_tensor = mat2_grad.value();
        outputs[1] = ttnn::moreh_matmul(
            input,
            output_grad,
            true,
            false,
            mat2_grad_tensor,
            std::nullopt,
            mat2_grad_mem_config,
            compute_kernel_config);
    }

    return outputs;
}
}  // namespace ttnn::operations::moreh::moreh_bmm_backward
