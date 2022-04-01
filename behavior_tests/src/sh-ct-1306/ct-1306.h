#include <cuda_runtime.h>
void getFlags(unsigned int *flags) {
  cudaHostGetFlags(flags, 0); // should give a warning
}
