// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "compute_kernel_api/tile_move_copy.h"
#include "dprint.h"

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t num_inner = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);
    constexpr bool gamma_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool beta_has_value = get_compile_time_arg_val(6) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(7) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(8) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(9) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(10) == 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    constexpr auto cb_x = tt::CB::c_in0;       // input
    constexpr auto cb_mask_w = tt::CB::c_in6;  // mask_w

    constexpr auto cb_xmm = tt::CB::c_intermed1;         // x - E[x]
    constexpr auto cb_xmm2 = tt::CB::c_intermed2;        // (x - E[x])^2

    constexpr uint32_t onetile = 1;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    // copy mask
    cb_wait_front(cb_mask_w, onetile);

    tile_regs_acquire();
    copy_tile_init();
    const uint32_t mask_dst = 8; // maybe this is problem, but why?
    copy_tile(cb_mask_w, first_tile, mask_dst);
    tile_regs_commit();

    tile_regs_wait();
    tile_regs_release();


    // multiplication
    cb_wait_front(cb_x, onetile);
    UNPACK(DPRINT << "ZZZZZZZZZZZZZZZ cb_x \n" << TSLICE(cb_x, 0, SliceRange{ .h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1 }) << "\n";)

    cb_reserve_back(cb_xmm, onetile);

    tile_regs_acquire();
    mul_tiles_init(cb_x, cb_x);
    mul_tiles(cb_x, cb_x, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, cb_xmm);
    tile_regs_release();

    cb_push_back(cb_xmm, onetile);

    // print value
    cb_wait_front(cb_xmm, onetile);
    UNPACK(DPRINT << "ZZZZZZZZZZZZZZZ cb_xmm2 \n" << TSLICE(cb_xmm, 0, SliceRange{ .h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1 }) << "\n";)
    cb_pop_front(cb_xmm, onetile);
}  // void MAIN
}  // namespace NAMESPACE
