// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_unreduced_backward_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "moreh_nll_loss_unreduced_backward.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {

void bind_moreh_nll_loss_unreduced_backward_operation(py::module &module) {
    bind_registered_operation(
        module,
        ttnn::moreh_nll_loss_unreduced_backward,
        R"doc(moreh_nll_loss_unreduced_backward(target_tensor: ttnn.Tensor, weight_tensor: Optional[ttnn.Tensor], output_grad_tensor: ttnn.Tensor, input_grad_tensor: Optional[ttnn.Tensor], ignore_index: int32, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[DeviceComputeKernelConfig]) -> ttnn.Tensor
            Compute backward for nll_loss operation with reduction set to None
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::moreh_nll_loss_unreduced_backward) &self,
               const Tensor &target_tensor,
               const std::optional<const Tensor> weight_tensor,
               const Tensor &output_grad_tensor,
               const std::optional<const Tensor> input_grad_tensor,
               const int32_t ignore_index,
               const std::optional<ttnn::MemoryConfig> &memory_config,
               std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) -> ttnn::Tensor {
                return self(
                    target_tensor,
                    weight_tensor,
                    output_grad_tensor,
                    input_grad_tensor,
                    ignore_index,
                    memory_config,
                    compute_kernel_config);
            },
            py::arg("target_tensor"),
            py::arg("weight_tensor"),
            py::arg("output_grad_tensor"),
            py::arg("input_grad_tensor"),
            py::arg("ignore_index"),
            py::arg("memory_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt});
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward
