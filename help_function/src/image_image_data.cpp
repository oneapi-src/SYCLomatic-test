// ====------ image_image_data.cpp---------- -*- C++ -* ----===////
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
  int i;
  // test_feature:image_data_type
  i = (int)dpct::image_data_type::matrix;
  i = (int)dpct::image_data_type::linear;
  i = (int)dpct::image_data_type::pitch;
  i = (int)dpct::image_data_type::unsupport;

  dpct::image_channel IC;
  sycl::range<3> Range(1,1,1);
  dpct::image_matrix IM(IC, Range);
  // test_feature:image_data
  dpct::image_data ID(&IM);

  // test_feature:get_data_type()
  auto DT = ID.get_data_type();

  // test_feature:set_data_type()
  ID.set_data_type(DT);

  void *data;
  // test_feature:set_data_ptr()
  ID.set_data_ptr(data);

  size_t T = 1;
  // test_feature:get_x()
  ID.get_x();

  // test_feature:set_x()
  ID.set_x(T);

  // test_feature:get_y()
  ID.get_y();

  // test_feature:set_y()
  ID.set_y(T);

  // test_feature:get_pitch()
  ID.get_pitch();

  // test_feature:set_pitch()
  ID.set_pitch(T);

  // test_feature:get_channel()
  IC = ID.get_channel();

  // test_feature:get_channel_data_type()
  auto ICDT = ID.get_channel_data_type();

  // test_feature:set_channel_data_type()
  ID.set_channel_data_type(ICDT);

  // test_feature:get_channel_size()
  T = ID.get_channel_size();

  // test_feature:get_channel_num()
  T = ID.get_channel_num();

  // test_feature:set_channel_num()
  ID.set_channel_num(T);

  // test_feature:get_channel_type()
  auto CT = ID.get_channel_type();

  // test_feature:set_channel_type()
  ID.set_channel_type(CT);

  return 0;
}
