#include <cassert>
#include <iostream>

#include <cuda.h>

__global__ void testFunnelShiftKernel(char * const TestResults) {
  TestResults[0] = (__funnelshift_l(0xAA000000, 0xBB, 8) == 0xBBAA);
  TestResults[1] = (__funnelshift_lc(0xAA000000, 0xBB, 16) == 0xBBAA00);
  TestResults[2] = (__funnelshift_r(0xAA00, 0xBB, 8) == 0xBB0000AA);
  TestResults[3] = (__funnelshift_rc(0xAA0000, 0xBB, 16) == 0xBB00AA);
}

int main() {
  constexpr int NumberOfTests = 4;
  char *TestResults;
  cudaMallocManaged(&TestResults, NumberOfTests * sizeof(*TestResults));
  testFunnelShiftKernel<<<1, 1>>>(TestResults);
  cudaDeviceSynchronize();
  for (int i = 0; i < NumberOfTests; i++) {
    if (TestResults[i] == 0) {
      std::cerr << "funnelshift test " << i << " failed" << std::endl;
    }
    assert(TestResults[i] != 0);
  }
}
