#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "header.dp.hpp"

#include <iostream>
#include <math.h>

#define EXPECTED_VALUE 8.0f

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  float *a, *b, *c, *d;
  int size = 1 << 20;

  a = sycl::malloc_shared<float>(size, q_ct1);
  b = sycl::malloc_shared<float>(size, q_ct1);
  c = sycl::malloc_shared<float>(size, q_ct1);
  d = sycl::malloc_shared<float>(size, q_ct1);

  for (int i=0; i<size; ++i)
    a[i] = b[i] = c[i] = d[i] = 2.0f;

  int threadsPerBlock = 256;
  int numOfBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // eval 2.0 * 2.0
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) {
        product(size, a, d, item_ct1);
      });

  // eval 4.0 + 2.0
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) {
        add(size, a, d, item_ct1);
      });

  // eval 6.0 - 2.0
  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) {
        subtract(size, a, d, item_ct1);
      });

  // eval 4.0 + 4.0
  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, numOfBlocks) *
                            sycl::range<3>(1, 1, threadsPerBlock),
                        sycl::range<3>(1, 1, threadsPerBlock)),
      [=](sycl::nd_item<3> item_ct1) {
        twice(size, d, item_ct1);
      });

  dev_ct1.queues_wait_and_throw();

  float error = 0.0f;
  for (int i=0; i<size; ++i)
    error = fmax(error, fabs(d[i] - EXPECTED_VALUE));

  // Expected value is "Error: 0" 
  std::cout << "Error: " << error << "\n";

  sycl::free(a, q_ct1);
  sycl::free(b, q_ct1);
  sycl::free(c, q_ct1);
  sycl::free(d, q_ct1);

  return error == 0.0f? 0 : 1;
}
