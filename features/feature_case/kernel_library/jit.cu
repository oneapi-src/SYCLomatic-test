#include "shared.hpp"

extern "C" __global__                     
void foo(int *a, int *b, int *c) {          
  int tid = blockIdx.x;
  
  if (tid<VEC_LENGTH) {
    a[tid] = b[tid] * c[tid] + SEED;
  }
}                                           
