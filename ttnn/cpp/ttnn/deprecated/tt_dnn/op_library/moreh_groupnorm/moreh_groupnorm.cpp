// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm/moreh_groupnorm_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {

inline uint32_t get_block_size(uint32_t num_tiles, uint32_t max_block_size) {
    uint32_t block_size{1};
    for (uint32_t current_block_size = max_block_size; current_block_size >= 1; current_block_size >>= 1) {
        if (num_tiles % current_block_size == 0) {
            block_size = current_block_size;
            break;
        }
    }
    return block_size;
}

}  // namespace

operation::ProgramWithCallbacks moreh_groupnorm_impl(
    const Tensor &input,
    uint32_t num_groups,
    float eps,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor &output,
    const std::optional<const Tensor> mean,
    const std::optional<const Tensor> rstd) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.get_legacy_shape();

    const auto n = input_shape[0];
    const auto c = input_shape[1];
    const auto h = input_shape[2];
    const auto w = input_shape[3];

    const auto origin_input_shape = input_shape.without_padding();

    const auto origin_h = origin_input_shape[2];
    const auto origin_w = origin_input_shape[3];

    const bool is_lastdim_layernorm = false;
    const bool is_groupnorm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_h % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_w % TILE_WIDTH : TILE_WIDTH;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const auto f_c = static_cast<float>(num_channels / num_groups);
    const auto f_ht = static_cast<float>(origin_h) / static_cast<float>(TILE_HEIGHT);
    const auto f_wt = static_cast<float>(origin_w) / static_cast<float>(TILE_WIDTH);
    auto scaler = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_c * f_ht * f_wt));

    const bool gamma_has_value = gamma.has_value();
    const bool beta_has_value = beta.has_value();
    const bool mean_has_value = mean.has_value();
    const bool rstd_has_value = rstd.has_value();

    const uint32_t block_size = get_block_size(num_inner_tiles, MAX_BLOCK_SIZE);

    std::cout << "HHHHHHHHHHHHHHH HOST block_size " << block_size << "\n";
    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_rows);

    log_debug(LogTest, fmt::format("num_cores_to_be_used: {}", num_cores_to_be_used).c_str());
    log_debug(LogTest, fmt::format("num_rows_per_core_group_1: {}", num_rows_per_core_group_1).c_str());
    log_debug(LogTest, fmt::format("num_rows_per_core_group_2: {}", num_rows_per_core_group_2).c_str());
    log_debug(LogTest, fmt::format("block_size: {}", block_size).c_str());

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner_tiles;  // input
    const uint32_t in6_t = 1;          // mask_w

    uint32_t im1_t = num_inner_tiles;  // x - E[x]
    uint32_t im2_t = 1;                // (x - E[x])^2

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    bool use_large_algorithm = false;

    std::cout << "HHHHHHHHHHHHHHH HOST num_inner_tiles " << num_inner_tiles << "\n";
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},  // input
            {CB::c_in6, in6_t},
            {CB::c_intermed1, im1_t},
            {CB::c_intermed2, im2_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file = use_large_algorithm ? "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm/"
                                                          "kernels/dataflow/reader_moreh_groupnorm_large.cpp"
                                                        : "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm/"
                                                          "kernels/dataflow/reader_moreh_groupnorm_small.cpp";

    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_groupnorm/kernels/dataflow/writer_moreh_groupnorm.cpp");

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_layernorm/kernels/moreh_layernorm_large_kernel.cpp"
            : "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_layernorm/kernels/moreh_layernorm_small_kernel.cpp";

    const std::vector<uint32_t> compute_args_group_1{
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        block_size,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_rows_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            block_size,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_rows_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();

    const auto output_addr = output.buffer()->address();
    const auto mean_addr = mean_has_value ? mean.value().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd.value().buffer()->address() : 0;

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            gamma_addr,
            static_cast<uint32_t>(is_dram(gamma)),
            static_cast<uint32_t>(gamma_has_value),
            beta_addr,
            static_cast<uint32_t>(is_dram(beta)),
            static_cast<uint32_t>(beta_has_value),
            *reinterpret_cast<uint32_t *>(&scaler),
            *reinterpret_cast<uint32_t *>(&eps),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            num_channels,
            origin_h,
            origin_w,
            block_size,
        };
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr,
            static_cast<uint32_t>(is_dram(output)),
            mean_addr,
            static_cast<uint32_t>(is_dram(mean)),
            static_cast<uint32_t>(mean_has_value),
            rstd_addr,
            static_cast<uint32_t>(is_dram(rstd)),
            static_cast<uint32_t>(rstd_has_value),
            tile_offset,
            num_rows_per_core,
            num_inner_tiles,
            num_groups,
            block_size,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           num_cores_to_be_used = num_cores_to_be_used,
                                           num_cores_y = num_cores_y](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        auto input_buffer = input_buffers.at(0);
        auto gamma_buffer = input_buffers.at(1);
        auto beta_buffer = input_buffers.at(2);

        auto ouput_buffer = output_buffers.at(0);
        auto mean_buffer = output_buffers.at(1);
        auto rstd_buffer = output_buffers.at(2);

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[0] = input_buffer->address();
                if (gamma_buffer != nullptr) {
                    runtime_args[2] = gamma_buffer->address();
                }
                if (beta_buffer != nullptr) {
                    runtime_args[5] = beta_buffer->address();
                }
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                runtime_args[0] = ouput_buffer->address();
                if (mean_buffer != nullptr) {
                    runtime_args[2] = mean_buffer->address();
                }
                if (rstd_buffer != nullptr) {
                    runtime_args[5] = rstd_buffer->address();
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
