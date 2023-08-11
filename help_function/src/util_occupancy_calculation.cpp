// ====--- util_calculate_max_active_wg_per_xecore.cpp ----- *- C++ -* ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {
  int num_blocks;
  int block_size = 128;
  size_t dynamic_shared_memory_size = 0;
  int sg_size = 32;
  bool used_barrier = true;
  bool used_large_grf = true;
  dpct::experimental::calculate_max_active_wg_per_xecore(
      &num_blocks, block_size,
      dynamic_shared_memory_size, sg_size, used_barrier,
      used_large_grf);

  int block_size_limit = 0;
  dpct::experimental::calculate_max_potential_wg(
      &num_blocks, &block_size, block_size_limit,
      dynamic_shared_memory_size, sg_size, used_barrier,
      used_large_grf);
  return 0;
}
