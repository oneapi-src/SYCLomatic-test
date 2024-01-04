#include "header.dp.hpp"

extern "C" SYCL_EXTERNAL void twice(int size, float *res,
                                    const sycl::nd_item<3> &item_ct1) {
  int blockIndex = item_ct1.get_group(2);
  int blockSize = item_ct1.get_local_range(2);
  int threadIndex = item_ct1.get_local_id(2);
  int index = blockIndex * blockSize + threadIndex;
  res[index] += res[index];
}

