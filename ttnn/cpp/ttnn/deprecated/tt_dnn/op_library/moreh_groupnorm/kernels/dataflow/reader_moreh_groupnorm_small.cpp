// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto gamma_addr = get_arg_val<uint32_t>(i++);
    const bool gamma_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool gamma_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto beta_addr = get_arg_val<uint32_t>(i++);
    const bool beta_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool beta_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto scaler = get_arg_val<uint32_t>(i++);
    const auto eps = get_arg_val<uint32_t>(i++);

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);
    const auto num_channels = get_arg_val<uint32_t>(i++);

    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);
    const auto block_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t onetile = 1;

    const auto Ht = (origin_h + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const auto Wt = (origin_w + TILE_WIDTH - 1) / TILE_WIDTH;

    const auto HtWt = Ht * Wt;

    const auto cb_id_input = tt::CB::c_in0;
    const auto cb_id_mask_w = tt::CB::c_in6;

    generate_mask_w(cb_id_mask_w, 1);

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    uint32_t input_tile_idx;

    DPRINT << "READER num_rows_per_core " << num_rows_per_core << "\n";
    DPRINT << "READER num_inner_tiles " << num_inner_tiles << "\n";

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        cb_reserve_back(cb_id_input, num_inner_tiles);
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            noc_async_read_tile(
                input_tile_idx, dram_input_addrg, input_l1_write_ptr + inner_idx * input_tile_bytes);
        }  // inner_idx loop
        noc_async_read_barrier();
        cb_push_back(cb_id_input, num_inner_tiles);

    }      // outer_idx loop

}  // void kernel_main()
