// ====------ thrust_system.cu------------------ *- CUDA -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

void isolated_foo() {
  thrust::error_code t1(static_cast<int>(0), thrust::generic_category());
  std::string arg;
  const char *char_arg = "test";
  int ev = 0;

  thrust::system::system_error test_1(t1, arg);
  thrust::system::system_error test_2(t1, char_arg);
  thrust::system::system_error test_3(t1);
  thrust::system::system_error test_4(ev, thrust::cuda_category(), arg);
  thrust::system::system_error test_5(ev, thrust::cuda_category(), char_arg);
  thrust::system::system_error test_6(ev, thrust::cuda_category());

  thrust::system_error test_7(t1, arg);
  thrust::system_error test_8(t1, char_arg);
  thrust::system_error test_9(t1);
  thrust::system_error test_10(ev, thrust::cuda_category(), arg);
  thrust::system_error test_11(ev, thrust::cuda_category(), char_arg);
  thrust::system_error test_12(ev, thrust::cuda_category());
  std::cout << "Caught Thrust error: " << test_1.what() << std::endl;
  std::cout << "Error code: " << test_1.code() << std::endl;
}

int main() {

  {
    thrust::error_code t1(static_cast<int>(0), thrust::generic_category());
    thrust::error_code t2(static_cast<int>(0), thrust::generic_category());
    bool ret1 = t1 != t2;
    bool ret2 = t1 == t2;
    bool ret3 = t1 < t2;

    if (ret1 || !ret2 || ret3) {
      printf("test1 failed!\n");
      exit(-1);
    }
    printf("test1 passed\n\n");
  }

  {
    thrust::error_code t1(static_cast<int>(0), thrust::generic_category());
    thrust::error_condition t2(0, thrust::generic_category());

    bool ret1 = t1 != t2;
    bool ret2 = t1 == t2;

    if (ret1 || !ret2) {
      printf("test2 failed!\n");
      exit(-1);
    }
    printf("test2 passed\n\n");
  }

  {
    thrust::error_code t1(static_cast<int>(0), thrust::generic_category());
    thrust::error_condition t2(0, thrust::generic_category());

    bool ret1 = t2 != t1;
    bool ret2 = t2 == t1;

    if (ret1 || !ret2) {
      printf("test3 failed!\n");
      exit(-1);
    }
    printf("test3 passed\n\n");
  }

  {
    thrust::error_condition t1(0, thrust::generic_category());
    thrust::error_condition t2(0, thrust::generic_category());
    bool ret = t1 != t2;

    bool ret1 = t1 != t2;
    bool ret2 = t1 == t2;
    bool ret3 = t1 < t2;

    if (ret1 || !ret2 || ret3) {
      printf("test4 failed!\n");
      exit(-1);
    }
    printf("test4 passed\n\n");
  }
  return 0;
}
