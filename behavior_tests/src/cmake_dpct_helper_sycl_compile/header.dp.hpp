#ifndef HEADERADDED
#define HEADERADDED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
extern "C" SYCL_EXTERNAL void product(int size, float *a, float *res,
                                      const sycl::nd_item<3> &item_ct1);

extern "C" SYCL_EXTERNAL void add(int size, float *a, float *res,
                                  const sycl::nd_item<3> &item_ct1);

extern "C" SYCL_EXTERNAL void subtract(int size, float *a, float *res,
                                       const sycl::nd_item<3> &item_ct1);

extern "C" SYCL_EXTERNAL void twice(int size, float *res,
                                    const sycl::nd_item<3> &item_ct1);
#endif
