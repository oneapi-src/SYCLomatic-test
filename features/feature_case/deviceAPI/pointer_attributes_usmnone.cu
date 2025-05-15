#include <cuda.h>



int main() {
  void *base_ptr;
  void *ptr;
  //CHECK:  if (DPCT_CHECK_ERROR(base_ptr = dpct::get_base_addr((dpct::device_ptr)ptr)) != 0);
  if (cuPointerGetAttribute(base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS);
}