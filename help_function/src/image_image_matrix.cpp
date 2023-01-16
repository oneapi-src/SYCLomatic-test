// ====------ image_image_matrix.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {

  dpct::image_channel IC;
  sycl::range<3> Range(1,1,1);
  // test_feature:image_matrix
  dpct::image_matrix IM(IC, Range);

  // test_feature:get_range()
  Range = IM.get_range();

  // test_feature:create_image()
  auto I = IM.create_image<3>();

  // test_feature:create_image(channel)
  I = IM.create_image<3>(IC);

  // test_feature:get_channel()
  auto IC2 = IM.get_channel();

  // test_feature:get_range()
  auto Range2 = IM.get_range();

  // test_feature:get_dims()
  auto Dim = IM.get_dims();

  // exit function auto destruct IM
  // test_feature:~image_matrix()
  return 0;
}
