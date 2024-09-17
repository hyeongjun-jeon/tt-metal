// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    uint32_t first_send_local_address = get_arg_val<uint32_t>(3);
    uint32_t second_send_local_address = get_arg_val<uint32_t>(4);
    uint32_t bytes_to_send = get_arg_val<uint32_t>(5);

    uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);

    volatile tt_l1_ptr uint32_t* first_local_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(first_send_local_address);
    volatile tt_l1_ptr uint32_t* second_local_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(second_send_local_address);

    volatile uint32_t* sem0 = reinterpret_cast<volatile uint32_t*>(get_semaphore(0));

    noc_semaphore_wait(sem0, 1);

    ncrisc_noc_fast_write_any_len(
        noc_index,
        NCRISC_WR_CMD_BUF,
        first_send_local_address,
        dst_noc_addr,
        bytes_to_send,      // size in bytes
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        false,  // linked
        1,      // num_dests 1 for non-mcast
        true    // multicast_path_reserve
    );

    // noc_async_write_barrier();

    ncrisc_noc_fast_write_any_len(
        noc_index,
        NCRISC_WR_REG_CMD_BUF, //NCRISC_WR_REG_CMD_BUF, --> when this matches above it passes
        second_send_local_address,
        dst_noc_addr,
        bytes_to_send,     // size in bytes
        NOC_UNICAST_WRITE_VC,
        false,  // mcasr
        false,  // linked
        1,      // num_dests 1 for non-mcast
        true    // multicast_path_reserve
    );

    noc_async_write_barrier();
}
