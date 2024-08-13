// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_device_operation.hpp"

namespace ttnn::operations::examples {

ExampleDeviceOperation::program_factory_t ExampleDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCore{};
}

void ExampleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void ExampleDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

ExampleDeviceOperation::shape_return_value_t ExampleDeviceOperation::compute_output_shapes(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return {tensor_args.input_tensor.tensor_attributes->shape, tensor_args.input_tensor.tensor_attributes->shape};
}

ExampleDeviceOperation::tensor_return_value_t ExampleDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto [output1_shape_opt, output2_shape_opt] = compute_output_shapes(operation_attributes, tensor_args);

    auto output1_shape = output1_shape_opt.value();
    auto output2_shape = output2_shape_opt.value();

    const auto& input_tensor = tensor_args.input_tensor;
    auto output1 = create_device_tensor(
        output1_shape,
        input_tensor.tensor_attributes->dtype,
        input_tensor.tensor_attributes->layout,
        input_tensor.device());

    auto output2 = create_device_tensor(
        output2_shape,
        input_tensor.tensor_attributes->dtype,
        input_tensor.tensor_attributes->layout,
        input_tensor.device());

    // return {output1, output2};
    return {output1, std::nullopt};
}


std::tuple<ExampleDeviceOperation::operation_attributes_t, ExampleDeviceOperation::tensor_args_t>
ExampleDeviceOperation::operator()(const Tensor& input_tensor) {
    return {
        operation_attributes_t{true, 42},
        tensor_args_t{input_tensor}
    };
}


}  // namespace ttnn::operations::examples
