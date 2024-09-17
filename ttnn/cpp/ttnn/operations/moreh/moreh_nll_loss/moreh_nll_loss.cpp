
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss.hpp"

#include <optional>

#include "moreh_nll_loss_step1/device/moreh_nll_loss_step1_device_operation.hpp"
#include "moreh_nll_loss_step2/device/moreh_nll_loss_step2_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss {

Tensor MorehNllLoss::invoke(
    const Tensor &input_tensor,
    const Tensor &target_tensor,
    const uint32_t reduction_mode,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const std::optional<const Tensor> output_tensor,
    const int32_t ignore_index,
    const std::optional<ttnn::MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    if (reduction_mode == MEAN) {
        TT_ASSERT(divisor_tensor.has_value());

        auto input_shape = input_tensor.get_legacy_shape();
        const uint32_t channel_size = input_shape[1];
        auto output_dtype = output_tensor.has_value() ? output_tensor.value().get_dtype() : input_tensor.get_dtype();

        const Tensor &step1_result = prim::moreh_nll_loss_step1(
            target_tensor,
            weight_tensor,
            ignore_index,
            reduction_mode,
            output_dtype,
            channel_size,
            memory_config,
            compute_kernel_config);

        ttnn::moreh_sum(
            step1_result, std::nullopt, false, divisor_tensor.value(), memory_config, compute_kernel_config);

        const Tensor &step2_result = prim::moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            reduction_mode,
            weight_tensor,
            divisor_tensor,
            output_tensor,
            ignore_index,
            memory_config,
            compute_kernel_config);
        return ttnn::moreh_sum(step2_result, std::nullopt, false, output_tensor, memory_config, compute_kernel_config);
    } else if (reduction_mode == SUM) {
        const Tensor &step2_result = prim::moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            reduction_mode,
            weight_tensor,
            std::nullopt,
            output_tensor,
            ignore_index,
            memory_config,
            compute_kernel_config);

        return ttnn::moreh_sum(step2_result, std::nullopt, false, output_tensor, memory_config, compute_kernel_config);
    }

    return prim::moreh_nll_loss_step2(
        input_tensor,
        target_tensor,
        reduction_mode,
        weight_tensor,
        std::nullopt,
        output_tensor,
        ignore_index,
        memory_config,
        compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss
