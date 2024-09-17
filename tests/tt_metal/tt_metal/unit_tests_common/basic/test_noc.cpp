// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/logger.hpp"


using namespace tt;

namespace unit_tests_common::noc {

void run_cmd_buffer_ordering_test(CommonFixture *fixture, tt_metal::Device *device, bool add_noc_traffic) {
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord sender_core = {0, 0};
    CoreCoord receiver_core = {3, 0};
    CoreCoord traffic_core0 = {1, 0};
    CoreCoord traffic_core1 = {2, 0};

    CoreCoord phys_sender_core = device->physical_core_from_logical_core(sender_core, CoreType::WORKER);
    CoreCoord phys_receiver_core = device->physical_core_from_logical_core(receiver_core, CoreType::WORKER);
    CoreCoord phys_traffic_core0 = device->physical_core_from_logical_core(traffic_core0, CoreType::WORKER);
    CoreCoord phys_traffic_core1 = device->physical_core_from_logical_core(traffic_core1, CoreType::WORKER);

    auto sender_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_order_sender.cpp",
        sender_core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_0});

    auto receiver_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/cmd_buf_order_receiver.cpp",
        receiver_core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_0});

    KernelHandle traffic_kernel0, traffic_kernel1;
    if (add_noc_traffic) {
        traffic_kernel0 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_l1.cpp",
            traffic_core0,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_0});

        traffic_kernel1 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_l1.cpp",
            traffic_core1,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_0});
    }

    uint32_t bytes_to_send = 4;
    uint32_t receiver_dst_address = L1_UNRESERVED_BASE;
    uint32_t first_sender_address = L1_UNRESERVED_BASE;
    uint32_t second_sender_address = align(first_sender_address + bytes_to_send, L1_ALIGNMENT);
    std::vector<uint32_t> sender_rt_args = {
        receiver_dst_address,
        (uint32_t)phys_receiver_core.x,
        (uint32_t)phys_receiver_core.y,
        first_sender_address,
        second_sender_address,
        bytes_to_send
    };

    uint32_t first_send_val = 0xDEADBEEF;
    uint32_t second_send_val = 0xABCDEF98;
    uint32_t receiver_result_address = align(receiver_dst_address + bytes_to_send, L1_ALIGNMENT);
    std::vector<uint32_t> receiver_rt_args = {
        (uint32_t)phys_sender_core.x,
        (uint32_t)phys_sender_core.y,
        first_send_val,
        second_send_val,
        receiver_dst_address,
        receiver_result_address
    };

    uint32_t traffic_num_to_send = 10;
    uint32_t traffic_single_send_bytes = 2048;
    uint32_t traffic_total_to_send_bytes = traffic_num_to_send * traffic_single_send_bytes;
    std::vector<uint32_t> traffic_core0_rt_args = {
        L1_UNRESERVED_BASE,
        (uint32_t)phys_traffic_core0.x,
        (uint32_t)phys_traffic_core0.y,
        L1_UNRESERVED_BASE,
        L1_UNRESERVED_BASE,
        (uint32_t)phys_traffic_core1.x,
        (uint32_t)phys_traffic_core1.y,
        traffic_num_to_send,
        traffic_single_send_bytes,
        traffic_total_to_send_bytes
    };

    std::vector<uint32_t> traffic_core1_rt_args = {
        L1_UNRESERVED_BASE,
        (uint32_t)phys_traffic_core1.x,
        (uint32_t)phys_traffic_core1.y,
        L1_UNRESERVED_BASE,
        L1_UNRESERVED_BASE,
        (uint32_t)phys_traffic_core0.x,
        (uint32_t)phys_traffic_core0.y,
        traffic_num_to_send,
        traffic_single_send_bytes,
        traffic_total_to_send_bytes
    };

    tt_metal::SetRuntimeArgs(program, sender_kernel, sender_core, sender_rt_args);
    tt_metal::SetRuntimeArgs(program, receiver_kernel, receiver_core, receiver_rt_args);
    if (add_noc_traffic) {
        tt_metal::SetRuntimeArgs(program, traffic_kernel0, traffic_core0, traffic_core0_rt_args);
        tt_metal::SetRuntimeArgs(program, traffic_kernel1, traffic_core1, traffic_core1_rt_args);
    }

    CreateSemaphore(program, CoreRange(sender_core, sender_core), 0, CoreType::WORKER);

    std::vector<uint32_t> first_sender_vals(bytes_to_send / sizeof(uint32_t), first_send_val);
    detail::WriteToDeviceL1(device, sender_core, first_sender_address, first_sender_vals, CoreType::WORKER);

    std::vector<uint32_t> second_sender_vals(bytes_to_send / sizeof(uint32_t), second_send_val);
    detail::WriteToDeviceL1(device, sender_core, second_sender_address, second_sender_vals, CoreType::WORKER);

    // clear out result space as well
    uint32_t receiver_l1_size_bytes = device->l1_size_per_core();
    std::vector<uint32_t> zero_vec((receiver_l1_size_bytes - receiver_dst_address)/sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(device, receiver_core, receiver_dst_address, zero_vec, CoreType::WORKER);

    tt::Cluster::instance().l1_barrier(device->id());

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromDeviceL1(device, receiver_core, receiver_result_address, sizeof(uint32_t), result_vec);

    std::cout << "Result " << std::hex << result_vec[0] << std::dec << std::endl;
    EXPECT_EQ(result_vec[0], 1) << "Receiver got results out of issue order";
}


}

// This test tries to validate that order of issuing commands is preserved when different command buffers are used with the same noc and static VC
// Only one sender and one receiver core are used
TEST_F(CommonFixture, CommandBufferOrdering) {
    unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), false);
}

// This test tries to validate that order of issuing commands is preserved when different command buffers are used with the same noc and static VC
// There is one sender and one receiver but an additional two cores that read from their local L1 and send to each other's L1
TEST_F(CommonFixture, CommandBufferOrderingAddNocTraffic) {
    unit_tests_common::noc::run_cmd_buffer_ordering_test(this, devices_.at(0), true);
}
