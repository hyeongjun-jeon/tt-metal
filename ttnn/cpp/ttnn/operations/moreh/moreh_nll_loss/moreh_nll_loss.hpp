
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss {

enum reduction_enum {
    NONE,
    SUM,
    MEAN,
};

struct MorehNllLoss {
    static Tensor invoke(
        const Tensor &input_tensor,
        const Tensor &target_tensor,
        const uint32_t reduction_mode,
        const std::optional<const Tensor> weight_tensor,
        const std::optional<const Tensor> divisor_tensor,
        const std::optional<const Tensor> output_tensor,
        const int32_t ignore_index,
        const std::optional<ttnn::MemoryConfig> &memory_config,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_nll_loss

namespace ttnn {
constexpr auto moreh_nll_loss = ttnn::
    register_operation_with_auto_launch_op<"ttnn::moreh_nll_loss", operations::moreh::moreh_nll_loss::MorehNllLoss>();
}  // namespace ttnn
