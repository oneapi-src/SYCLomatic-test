// ====------ kernel_function_win.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <libloaderapi.h>
#include <string>
#include <iostream>
template<class T>
void testTemplateKernel(T *data)
{
}

void testKernel(void* data)
{
}

template<class T>
int getTemplateFuncAttrs()
{
  //test_feature:kernel_function_info
  dpct::kernel_function_info attrs;
  //test_feature:get_kernel_function_info
  dpct::get_kernel_function_info(&attrs, (const void *)testTemplateKernel<T>);

  int threadPerBlock = attrs.max_work_group_size;

  return threadPerBlock;
}

int getFuncAttrs()
{
  //test_feature:kernel_function_info
  dpct::kernel_function_info attrs;
  //test_feature:get_kernel_function_info
  dpct::get_kernel_function_info(&attrs, (const void *)testKernel);

  int threadPerBlock = attrs.max_work_group_size;

  return threadPerBlock;
}

int main(int argc, char *argv[])
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  int Size = dev_ct1.get_info<cl::sycl::info::device::max_work_group_size>();
  if(getTemplateFuncAttrs<int>() != Size || getFuncAttrs() != Size) {
    std::cout << "dpct::get_kernel_function_info verify failed" << std::endl;
    return -1;
  }

  HMODULE M;
  //test_feature:kernel_functor
  dpct::kernel_functor F;

  M = LoadLibraryA(argv[1]);

  std::string FunctionName = "foo_wrapper";
  F = (dpct::kernel_functor)GetProcAddress(M, FunctionName.c_str());

  int sharedSize = 10;
  void **param = nullptr, **extra = nullptr;

  int *dev = sycl::malloc_shared<int>(16, q_ct1);
  for(int i = 0; i < 16; i++) {
      dev[i] = 0;
  }
  param = (void **)(&dev);
  F(q_ct1, sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)), sharedSize, param, extra);
  q_ct1.wait_and_throw();
  
  bool Pass = true;
  for(int i = 0; i < 16; i++) {
      if(dev[i] != i) {
          Pass = false;
          break;
      }
  }
  if(!Pass) {
    std::cout << "dpct::kernel_functor verify failed" << std::endl;
    sycl::free(dev, q_ct1);
    FreeLibrary(M);
    return 1;
  } else {
    std::cout << "pass" << std::endl;
    sycl::free(dev, q_ct1);
    FreeLibrary(M);
    return 0;
  }
}
