// ====------ image_image_channel.cpp---------- -*- C++ -* ----===////
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
  // test_feature:image_channel_data_type
  int i = (int)dpct::image_channel_data_type::signed_int;
  i = (int)dpct::image_channel_data_type::unsigned_int;
  i = (int)dpct::image_channel_data_type::fp;


  // test_feature:image_channel
  dpct::image_channel IC;

  // test_feature:get_channel_data_type()
  auto ICDT = IC.get_channel_data_type();

  // test_feature:set_channel_data_type()
  IC.set_channel_data_type(ICDT);

  // test_feature:get_total_size()
  IC.get_total_size();

  // test_feature:get_channel_num()
  auto CN = IC.get_channel_num();

  // test_feature:set_channel_num()
  IC.set_channel_num(CN);

  // test_feature:get_channel_type()
  auto CT = IC.get_channel_type();

  // test_feature:set_channel_type()
  IC.set_channel_type(CT);

  // test_feature:get_channel_order()
  auto CO = IC.get_channel_order();

  // test_feature:get_channel_size()
  auto CS = IC.get_channel_size();

  return 0;
}
