// ====------ thrust_test_device_ptr_2.cpp---------- -*- C++ -* ----===////
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
  dpcpp -fno-sycl-unnamed-lambda thrust_test_device_ptr_2.cpp

run:
  ./a.out

expected output:
Passed

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
static void dumpMaps(const char *name, int *maps, int num) {
  std::cout << name << "\n";
  for (int i = 0; i < num; ++i) {
    std::cout << i << ": " << maps[i] << "\n";
  }
}

int main(void) {
  const int numsH = 10;
  const int valuep1 = -1;
  const int valuepkey = -2;
  const int valuepval = -3;

  std::vector<int> mapsp1H(numsH);
  std::vector<int> mapspkeyH(numsH);
  std::vector<int> mapspvalH(numsH);

  std::fill(mapsp1H.begin(), mapsp1H.begin() + numsH, valuep1);
  std::fill(mapspkeyH.begin(), mapspkeyH.begin() + numsH, valuepkey);
  std::fill(mapspvalH.begin(), mapspvalH.begin() + numsH, valuepval);
  // dumpMaps("mapsp1H", mapsp1H, numsH);
  // dumpMaps("mapspkeyH", mapspkeyH, numsH);
  // dumpMaps("mapspvalH", mapspvalH, numsH);

  // cudaMalloc
  int *mapsp1D = (int *)sycl::malloc_device(numsH * sizeof(int),
                                            dpct::get_current_device(),
                                            dpct::get_default_context());
  int *mapspkeyD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                              dpct::get_current_device(),
                                              dpct::get_default_context());
  int *mapspvalD = (int *)sycl::malloc_device(numsH * sizeof(int),
                                              dpct::get_current_device(),
                                              dpct::get_default_context());

  // cudaMemcpy
  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t1>(
                dpct::get_default_queue()),
            mapsp1H.begin(), mapsp1H.begin() + numsH, mapsp1D);
  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t2>(
                dpct::get_default_queue()),
            mapspkeyH.begin(), mapspkeyH.begin() + numsH, mapspkeyD);
  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t3>(
                dpct::get_default_queue()),
            mapspvalH.begin(), mapspvalH.begin() + numsH, mapspvalD);

  // snapshot from Pennant
  dpct::device_pointer<int> mapsp1T(mapsp1D);
  dpct::device_pointer<int> mapspkeyT(mapspkeyD);
  dpct::device_pointer<int> mapspvalT(mapspvalD);
  // dumpMaps("mapsp1T", mapsp1T, numsH);
  // dumpMaps("mapspkeyT", mapspkeyT, numsH);
  // dumpMaps("mapspvalT", mapspvalT, numsH);

  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_7>(
                dpct::get_default_queue()),
            mapsp1T, mapsp1T + numsH, mapspkeyT);
  dpct::iota(oneapi::dpl::execution::make_device_policy<class Policy_8>(
                 dpct::get_default_queue()),
             mapspvalT, mapspvalT + numsH);
  // dumpMaps("mapspkeyT after copy", mapspkeyT, numsH);
  // dumpMaps("mapspvalT after sequence", mapspvalT, numsH);

  std::vector<int> mapspkeyHH(numsH);
  std::vector<int> mapspvalTH(numsH);

  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t4>(
                dpct::get_default_queue()),
            mapsp1T, mapsp1T + numsH, mapspkeyHH.begin());

  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_t5>(
                dpct::get_default_queue()),
            mapspvalT, mapspvalT + numsH, mapspvalTH.begin());

  dpct::get_default_queue().wait();
  bool pass = true;
  for (int i = 0; i < numsH; ++i) {
    if (mapspkeyHH[i] != valuep1) {
      std::cout << "Unexpected key: mapspkeyT[" << i << "] == " << mapspkeyHH[i]
                << ", expected " << valuep1 << "\n";
      pass = false;
    }
    if (mapspvalTH[i] != i) {
      std::cout << "Unexpected val: mapspvalT[" << i << "] == " << mapspvalTH[i]
                << ", expected " << i << "\n";
      pass = false;
    }
  }
  std::cout << std::endl << (pass ? "Passed" : "Failed") << "\n";

  return 0;
}
