extern "C" __global__                     
void foo(int *a, int *b, int *c) {          
  int tid = blockIdx.x;
  
  if (tid<128) {
    a[tid] = b[tid] * c[tid] + 59;
  }
}                                           
