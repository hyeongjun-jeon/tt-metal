
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss.hpp"

#include <optional>

#include "moreh_nll_loss_step2/device/moreh_nll_loss_step2_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss {

Tensor MorehNllLoss::invoke(
    const Tensor &input_tensor,
    const Tensor &target_tensor,
    const bool reduction_mean,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const std::optional<const Tensor> output_tensor,
    const int32_t ignore_index,
    const std::optional<ttnn::MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    const Tensor &step2_result = prim::moreh_nll_loss_step2(
        input_tensor,
        target_tensor,
        reduction_mean,
        weight_tensor,
        std::nullopt,
        output_tensor,
        ignore_index,
        memory_config,
        compute_kernel_config);

    return ttnn::moreh_sum(step2_result, std::nullopt, false, output_tensor, memory_config, compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss
