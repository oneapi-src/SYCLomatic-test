// ====------ image_sampling_info.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {
  dpct::sampling_info SI;
  // test_feature:get_addressing_mode()
  auto AM = SI.get_addressing_mode();

  // test_feature:set(addressing_mode)
  SI.set(AM);

  // test_feature:get_filtering_mode()
  auto FM = SI.get_filtering_mode();

  // test_feature:set(filtering_mode)
  SI.set(FM);

  // test_feature:get_coordinate_normalization_mode()
  auto CNM = SI.get_coordinate_normalization_mode();

  // test_feature:set(coordinate_normalization_mode)
  SI.set(CNM);

  // test_feature:is_coordinate_normalized()
  auto ICN = SI.is_coordinate_normalized();

  // test_feature:set_coordinate_normalization_mode()
  SI.set_coordinate_normalization_mode(ICN);

  // test_feature:set(addressing_mode, filtering_mode, coordinate_normalization_mode)
  SI.set(AM, FM, CNM);

  // test_feature:set(addressing_mode, filtering_mode, is_normalized)
  SI.set(AM, FM, 0);

  // test_feature:get_sampler()
  auto SA = SI.get_sampler();

  return 0;
}
