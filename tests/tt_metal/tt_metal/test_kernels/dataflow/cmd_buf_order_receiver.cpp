// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_noc_x = get_arg_val<uint32_t>(0);
    uint32_t src_noc_y = get_arg_val<uint32_t>(1);
    uint32_t first_expected_val  = get_arg_val<uint32_t>(2);
    uint32_t second_expected_val  = get_arg_val<uint32_t>(3);
    uint32_t receive_local_address = get_arg_val<uint32_t>(4);
    uint32_t result_address = get_arg_val<uint32_t>(5);

    uint32_t noc_writes_issued_address = result_address + L1_ALIGNMENT;
    uint32_t src_noc_xy = uint32_t(NOC_XY_ENCODING(src_noc_x, src_noc_y));

    volatile tt_l1_ptr uint32_t* local_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_local_address);

    uint32_t offset = (noc_index << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(NIU_MST_WR_ACK_RECEIVED);
    uint64_t src_noc_wr_issued_addr = get_noc_addr_helper(src_noc_xy, offset);

    ncrisc_noc_fast_read_any_len(noc_index, NCRISC_RD_CMD_BUF, src_noc_wr_issued_addr, noc_writes_issued_address, 4);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* noc_writes_issued_address_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_writes_issued_address);
    uint32_t initial_writes_acked = *noc_writes_issued_address_ptr;

    DPRINT << " noc from other core " << initial_writes_acked << ENDL();

    noc_semaphore_inc(get_noc_addr_helper(src_noc_xy, get_semaphore(0)), 1);

    bool received_first = false;
    bool received_second = false;
    uint32_t received = 0;
    do {
        received = *local_address;
        if (received == first_expected_val) {
            received_first = true;
        }
        if (received == second_expected_val) {
            received_second = true;
        }
        invalidate_l1_cache();
        // DPRINT << "received " << HEX() << received << ENDL();
    } while (not received_first and not received_second); // got one value


    uint32_t total_writes_acked = initial_writes_acked;
    while (total_writes_acked != (initial_writes_acked + 2)) {
        ncrisc_noc_fast_read_any_len(noc_index, NCRISC_RD_CMD_BUF, src_noc_wr_issued_addr, noc_writes_issued_address, 4);
        noc_async_read_barrier();
        total_writes_acked = *noc_writes_issued_address_ptr;
    }

    DPRINT << " noc from other core " << total_writes_acked << ENDL();

    uint32_t order_check = (*local_address == second_expected_val) ? 1 : 0xBADC00DE;

    volatile tt_l1_ptr uint32_t* result_address_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_address);
    *result_address_ptr = order_check;
}
