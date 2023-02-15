//__constant__ array must be migrated to a static variable
__constant__ int dvals[2];

void init_l1(int *hvals) {
  cudaMemcpyToSymbol(dvals, hvals, 2 * sizeof(int), 0, cudaMemcpyHostToDevice);
}

void get_l1(int *target) {
  cudaMemcpyFromSymbol(target, dvals, 2 * sizeof(int), 0, cudaMemcpyDeviceToHost);
}
