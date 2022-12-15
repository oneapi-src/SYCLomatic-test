#include <iostream>
#include <cuda.h>
#include "init.hpp"

static CUdevice   device;
static CUcontext  context;

void init_CUDA() {
  int cnt = 0;
  CUresult err = cuInit(0);

  if (err==CUDA_SUCCESS)
    checkErrors(cuDeviceGetCount(&cnt));

  if (cnt==0) {
    fprintf(stderr, "Error: no CUDA devices\n");
    exit(-1);
  }

  // use first CUDA device
  checkErrors(cuDeviceGet(&device, 0));

  err = cuCtxCreate(&context, 0, device);
  if (err!=CUDA_SUCCESS) {
      fprintf(stderr, "Initialization failure.\n");
      exit(-1);
  }  
}
