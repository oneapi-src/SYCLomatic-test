#ifndef HEADERADDED
#define HEADERADDED
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

void add_wrapper(int, float *, float *, int, int, sycl::queue &);
void product_wrapper(int, float *, float *, int, int, sycl::queue &);
void subtract_wrapper(int, float *, float *, int, int, sycl::queue &);
void twice_wrapper(int, float *, int, int, sycl::queue &);
#endif
