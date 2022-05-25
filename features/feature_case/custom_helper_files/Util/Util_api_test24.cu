// ====------ Util_api_test24.cu ------------------------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// TEST_FEATURE: Util_logical_group_get_local_linear_range

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ void foo() {
  cg::thread_block ttb = cg::this_thread_block();
  cg::thread_block_tile<8> tbt = cg::tiled_partition<8>(ttb);
  tbt.size();
}