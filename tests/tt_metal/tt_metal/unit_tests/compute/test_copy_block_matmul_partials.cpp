// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"


using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::matmul_partials {

struct CopyBlockMatmulPartialsConfig {
    uint32_t single_tile_size;
    uint32_t num_tiles;
    uint32_t reader_ublock;
    uint32_t writer_ublock;
    uint32_t compute_ublock;
    uint32_t src0_cb_index;
    uint32_t ouput_cb_index;
    bool fp32_dest_acc = false;
};

void run_single_core_copy_block_matmul_partials(tt_metal::Device* device, const CopyBlockMatmulPartialsConfig& test_config) {


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};
    uint32_t single_tile_size = test_config.single_tile_size;
    uint32_t num_tiles = test_config.num_tiles;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
                                    .device=device,
                                    .size = dram_buffer_size,
                                    .page_size = dram_buffer_size,
                                    .buffer_type = tt_metal::BufferType::DRAM
                                    };

    auto src_dram_buffer_bf16 = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer_bf16->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    auto dram_src_noc_xy = src_dram_buffer_bf16->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = test_config.src0_cb_index;
    uint32_t num_input_tiles = test_config.reader_ublock;

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);

    if (test_config.fp32_dest_acc) {
        cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float32}})
            .set_page_size(src0_cb_index, single_tile_size);
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = test_config.ouput_cb_index;
    uint32_t num_output_tiles = test_config.writer_ublock;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, single_tile_size);
    if (test_config.fp32_dest_acc) {
        cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float32}})
            .set_page_size(ouput_cb_index, single_tile_size);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_pop_n.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles), // total tiles to transfer
        uint(test_config.compute_ublock), // tiles to transfer in a single iteration/copy_block call
        uint(src0_cb_index), // Input CB idx
        uint(ouput_cb_index) // Output CB idx
    };


    std::map<string, string> defines;
    if (test_config.fp32_dest_acc) {
        defines["DST_ACCUM_MODE"] = "1";
    }
    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block_matmul_partials.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = defines}
    );


    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, 0);

    if (test_config.fp32_dest_acc) {
        auto src_vec_float = generate_uniform_random_vector<float>(
            -100, 100, dram_buffer_size/sizeof(float), 0);
        for (auto i = 0; i < src_vec.size(); i++) {
            std::memcpy(&src_vec[i], &src_vec_float[i], sizeof(float));
            src_vec[i] &= 0xFFFFE000;
        }
    }

    tt_metal::detail::WriteToBuffer(src_dram_buffer_bf16, src_vec);

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel,
        core,
        {dram_buffer_src_addr,
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        num_tiles,
        src0_cb_index,
        test_config.reader_ublock,
        false});

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {dram_buffer_dst_addr,
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        num_tiles,
        ouput_cb_index,
        test_config.writer_ublock,
        false});

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result_vec_bf16;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec_bf16);


    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////
    EXPECT_EQ(src_vec.size(), result_vec_bf16.size());
    EXPECT_EQ(src_vec, result_vec_bf16);
    uint32_t cnt = 0;
    for (auto i = 0; i < src_vec.size(); i++) {
        if (src_vec[i] != result_vec_bf16[i] && cnt < 5) {
            cnt++;
            std::cout << "src_vec[" << std::dec << i << "] = " << std::hex << src_vec[i] << std::endl;
            std::cout << "result_vec_bf16[" << std::dec << i << "] = " << std::hex << result_vec_bf16[i] << std::endl;
        }
    }


}
} // namespace unit_tests::compute::matmul_partials

////////////////////////////////////////////////////////////////////////////
//                             Tests
// ------------------------------------------------------------------------
// These tests aim to cover usage of these calls:
// - copy_block_matmul_partials
// - matmul_pack_tile
//
// Tests which contain a string in RXWYCZ format in their name cover
// different scenarios in reader/writer/compute kernel usage. Letters
// R, W and C represent reader, writer and compute kernel, respectively,
// while the numbers X, Y and Z represent how many tiles will a kernel
// move in a single loop iteration. This is important because depending
// on these numbers, synchronization points are met at different places.
// Since there can be a maximum of 8 32-by-32 tiles in DEST reg when using
// half of it (for MATH/PACK sync purporses), highest bandwidth is achieved
// when all three parameters are 8. It's also possible to enforce MATH/PACK
// serialization by telling writer to wait for a single tile to be avail-
// able in output CB.
//
////////////////////////////////////////////////////////////////////////////
TEST_F(DeviceFixture, ComputeCopyBlockMatmulPartialsR8W8C8) {
    for (bool fp32_dest_acc : {true, false}) {
        log_info(LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
        unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
            .single_tile_size = 2 * 1024,
            .num_tiles = 8,
            .reader_ublock = 8,
            .writer_ublock = 8,
            .compute_ublock = 8,
            .src0_cb_index = 0,
            .ouput_cb_index = 16,
            .fp32_dest_acc = fp32_dest_acc
        };
        unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(this->devices_.at(0), test_config);
        log_info(LogTest, "Passed.");
    }
}

TEST_F(DeviceFixture, ComputeCopyBlockMatmulPartialsR8W8C1) {
    for (bool fp32_dest_acc : {true, false}) {
        log_info(LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
            unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
            .single_tile_size = 2 * 1024,
            .num_tiles = 8,
            .reader_ublock = 8,
            .writer_ublock = 8,
            .compute_ublock = 1,
            .src0_cb_index = 0,
            .ouput_cb_index = 16
        };
        unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(this->devices_.at(0), test_config);
        log_info(LogTest, "Passed.");
    }
}

TEST_F(DeviceFixture, ComputeCopyBlockMatmulPartialsR8W1C1) {
    for (bool fp32_dest_acc : {true, false}) {
        log_info(LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
        unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
            .single_tile_size = 2 * 1024,
            .num_tiles = 8,
            .reader_ublock = 8,
            .writer_ublock = 1,
            .compute_ublock = 1,
            .src0_cb_index = 0,
            .ouput_cb_index = 16
        };
        unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(this->devices_.at(0), test_config);
        log_info(LogTest, "Passed.");
    }
}

TEST_F(DeviceFixture, ComputeCopyBlockMatmulPartialsR1W1C1) {
    for (bool fp32_dest_acc : {true, false}) {
        log_info(LogTest, "FP32 Dest Acc is {}", fp32_dest_acc ? "true." : "false.");
        unit_tests::compute::matmul_partials::CopyBlockMatmulPartialsConfig test_config = {
            .single_tile_size = 2 * 1024,
            .num_tiles = 1,
            .reader_ublock = 1,
            .writer_ublock = 1,
            .compute_ublock = 1,
            .src0_cb_index = 0,
            .ouput_cb_index = 16
        };
        unit_tests::compute::matmul_partials::run_single_core_copy_block_matmul_partials(this->devices_.at(0), test_config);
        log_info(LogTest, "Passed.");
    }
}
