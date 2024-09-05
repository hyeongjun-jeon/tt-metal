// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(ARCH_WORMHOLE_B0)

#define COMPILE_FOR_ERISC

#include "llrt/hal.hpp"
#include "llrt/wormhole/wh_hal.hpp"
#include "hw/inc/wormhole/dev_mem_map.h"
#include "hw/inc/wormhole/eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"
#include "hw/inc/dev_msgs.h"

#define GET_ETH_MAILBOX_ADDRESS_HOST(x) \
    ((uint64_t) & (((mailboxes_t *)eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE)->x))

namespace tt {

namespace tt_metal {

HalCoreInfoType create_active_eth_mem_map() {

    constexpr uint32_t num_proc_per_active_eth_core = 1;

    std::vector<DeviceAddr> mem_map_bases;
    mem_map_bases.resize(static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::COUNT));
    mem_map_bases[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::BARRIER)] = MEM_L1_BARRIER;
    mem_map_bases[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::LAUNCH)] = GET_ETH_MAILBOX_ADDRESS_HOST(launch);
    mem_map_bases[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::WATCHER)] = GET_ETH_MAILBOX_ADDRESS_HOST(watcher);
    mem_map_bases[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::DPRINT)] = GET_ETH_MAILBOX_ADDRESS_HOST(dprint_buf);
    mem_map_bases[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::PROFILER)] = GET_ETH_MAILBOX_ADDRESS_HOST(profiler);
    mem_map_bases[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::KERNEL_CONFIG)] = eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_BASE;
    mem_map_bases[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::UNRESERVED)] = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    std::vector<uint32_t> mem_map_sizes;
    mem_map_sizes.resize(static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::COUNT));
    mem_map_sizes[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::BARRIER)] = sizeof(uint32_t);
    mem_map_sizes[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::LAUNCH)] = sizeof(launch_msg_t);
    mem_map_sizes[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::WATCHER)] = sizeof(watcher_msg_t);
    mem_map_sizes[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::DPRINT)] = sizeof(dprint_buf_msg_t);
    mem_map_sizes[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::PROFILER)] = sizeof(profiler_msg_t);
    mem_map_sizes[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::KERNEL_CONFIG)] = eth_l1_mem::address_map::ERISC_L1_KERNEL_CONFIG_SIZE;
    mem_map_sizes[static_cast<std::underlying_type<HalMemAddrType>::type>(HalMemAddrType::UNRESERVED)] = eth_l1_mem::address_map::MAX_SIZE - eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    return {HalProgrammableCoreType::ACTIVE_ETH, CoreType::ETH, num_proc_per_active_eth_core, mem_map_bases, mem_map_sizes};
}

}  // namespace tt_metal
}  // namespace tt
#endif
