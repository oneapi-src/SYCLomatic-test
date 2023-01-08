// ====------ thrust_test-pennet_simple_pstl.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
/*
Thrust test case:

Environment setup:
  oneAPI environment: dpcpp, dpct, and tbb

build:
  dpcpp -fno-sycl-unnamed-lambda thrust_test-pennet_simple_pstl.cpp

run:
  ./a.out

expected output:
i = 0, 101 9;
i = 1, 102 8;
i = 2, 103 7;
i = 3, 104 6;
i = 4, 105 5;
i = 5, 106 4;
i = 6, 107 3;
i = 7, 108 2;
i = 8, 109 1;
i = 9, 110 0;

done
*/

#define DPCT_NAMED_LAMBDA

#include <cstdio>

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

static void dumpMaps(const char *name, dpct::device_pointer<int> &maps,
                     int num) {
  std::cout << name << "\n";
  for (int i = 0; i < num; ++i) {
    std::cout << i << ": " << maps[i] << "\n";
  }
}

int main() {

  int *mapsp1D, *mapspkeyD, *mapspvalD;
  int numsH = 10;

  mapsp1D = (int *)sycl::malloc_device(numsH * sizeof(int),
                                       dpct::get_current_device(),
                                       dpct::get_default_context());
  mapspkeyD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                         dpct::get_current_device(),
                                         dpct::get_default_context());
  mapspvalD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                         dpct::get_current_device(),
                                         dpct::get_default_context());

  dpct::device_pointer<int> mapsp1T(mapsp1D);
  dpct::device_pointer<int> mapspkeyT(mapspkeyD);
  dpct::device_pointer<int> mapspvalT(mapspvalD);

  std::vector<int> value(numsH);
  for (int i = 0; i < numsH; ++i) {
    value[i] = 100 + numsH - i;
  }

  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t3>(
                dpct::get_default_queue()),
            value.begin(), value.begin() + numsH, mapsp1T);

  // copy vector: mapsp1T -> mapspkeyT
  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_44009e>(
                dpct::get_default_queue()),
            mapsp1T, mapsp1T + numsH, mapspkeyT);
  // dumpMaps("mapspkeyT after copy", mapspkeyT, numsH);

  // create a sequence of numbers in mapspvalT vector [0, 1, 2, ..., 9]
  dpct::iota(oneapi::dpl::execution::make_device_policy<class Policy_9a9f11>(
                 dpct::get_default_queue()),
             mapspvalT, mapspvalT + numsH);
  // dumpMaps("mapspvalT after sequence", mapspvalT, numsH);

  // sort both mapspkeyT and mapspvalT, so that the elements in mapspkeyT are in
  // smallest first order
  dpct::stable_sort(
      oneapi::dpl::execution::make_device_policy<class Policy_3b8d2e>(
          dpct::get_default_queue()),
      mapspkeyT, mapspkeyT + numsH, mapspvalT);
  dpct::get_default_queue().wait();

  std::vector<int> mapspkeyHH(numsH);
  std::vector<int> mapspvalTH(numsH);

  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t4>(
                dpct::get_default_queue()),
            mapspkeyT, mapspkeyT + numsH, mapspkeyHH.begin());

  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t5>(
                dpct::get_default_queue()),
            mapspvalT, mapspvalT + numsH, mapspvalTH.begin());

  bool pass = true;
  for (int i = 0; i < numsH; ++i) {
    if ((mapspkeyHH[i] != i + 101) || (mapspvalTH[i] != 9 - i))
      pass = false;
    std::cout << "i = " << i << ", " << mapspkeyHH[i] << " " << mapspvalTH[i]
              << ";\n";
  }
  std::cout << std::endl << (pass ? "Passed" : "Failed") << "\n";
}
