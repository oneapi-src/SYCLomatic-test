// ====------ driverDevice.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUM 1
#define CUDA_SAFE_CALL( call) do {\
  int err = call;                \
} while (0)
int main(){
  int result1, result2;
  int *presult1 = &result1, *presult2 = &result2;
  CUdevice device, peerDevice;
  CUdevice *pdevice = &device;
  cuDeviceGet(&device, 0);
  cuDeviceGet(&device, NUM);
  cuDeviceGet(pdevice, 0);
  cuDeviceGet((CUdevice *)pdevice, 0);
  CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_INTEGRATED, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device);
  CUDA_SAFE_CALL(cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);

  cuDeviceComputeCapability(&result1, &result2, device);
  CUDA_SAFE_CALL(cuDeviceComputeCapability(&result1, &result2, device));
  CUDA_SAFE_CALL(cuDeviceGetCount(&result1));
  cuDeviceGetCount(&result1);
  CUDA_SAFE_CALL(cuDeviceGetCount(&result1));

  char name[100];
  cuDeviceGetName(name, 90, device);
  CUDA_SAFE_CALL(cuDeviceGetName(name, 90, device));

  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
  cuDeviceGetAttribute(&result1, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);

  CUdevice_P2PAttribute p2p_attr = CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED;
  cuDeviceGetP2PAttribute(&result1, p2p_attr, device, peerDevice);

  cuDeviceGetP2PAttribute(&result1, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, device, peerDevice);
  cuDeviceGetP2PAttribute(&result1, CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, device, peerDevice);
#if (CUDA_VERSION <= 10000)
  cuDeviceGetP2PAttribute(&result1, CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED, device, peerDevice);
#else
  cuDeviceGetP2PAttribute(&result1, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED, device, peerDevice);
#endif
#if (CUDA_VERSION != 9020)
  cuDeviceGetP2PAttribute(&result1, CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED, device, peerDevice);
#endif

  CUcontext context;
  unsigned int flags = CU_CTX_MAP_HOST;
  flags += CU_CTX_SCHED_BLOCKING_SYNC;
  flags += CU_CTX_SCHED_SPIN;

  if (cuCtxCreate(&context, flags, device) == CUDA_SUCCESS) {
    return 0;
  }

  cuDevicePrimaryCtxSetFlags(device, flags);

  int active;
  cuDevicePrimaryCtxGetState(device, &flags, &active);

  cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);

  cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 10);
  size_t printfsize;

  cuCtxGetLimit(&printfsize, CU_LIMIT_PRINTF_FIFO_SIZE);
  return 0;
}

