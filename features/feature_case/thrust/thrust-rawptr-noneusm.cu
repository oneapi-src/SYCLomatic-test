// ====------ thrust-rawptr-noneusm.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include "report.h"
// for cuda 12.0
#include <thrust/partition.h>
#include <thrust/unique.h>


struct greater_than_zero
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x > 0;
  }
  typedef int argument_type;
};


int main(){
  greater_than_zero pred;

  float *host_ptr_A;
  float *host_ptr_R;
  float *host_ptr_S;
  float *device_ptr_A;
  float *device_ptr_S;
  float *device_ptr_R;


  // replace_if
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::replace_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, pred, 0);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("replace_if", host_ptr_R[0], -5);
  Report::check("replace_if", host_ptr_R[1], 0);
  Report::check("replace_if", host_ptr_R[2], 0);
  Report::check("replace_if", host_ptr_R[3], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::replace_if(host_ptr_A, host_ptr_A+10, pred, 0);
  Report::check("replace_if", host_ptr_A[0], -5);
  Report::check("replace_if", host_ptr_A[1], 0);
  Report::check("replace_if", host_ptr_A[2], 0);
  Report::check("replace_if", host_ptr_A[3], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::replace_if(thrust::device, device_ptr_A, device_ptr_A+10, pred, 0);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("replace_if", host_ptr_R[0], -5);
  Report::check("replace_if", host_ptr_R[1], 0);
  Report::check("replace_if", host_ptr_R[2], 0);
  Report::check("replace_if", host_ptr_R[3], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::replace_if(host_ptr_A, host_ptr_A+10, host_ptr_S, pred, 0);
  Report::check("replace_if", host_ptr_A[0], -5);
  Report::check("replace_if", host_ptr_A[1], 0);
  Report::check("replace_if", host_ptr_A[2], 0);
  Report::check("replace_if", host_ptr_A[3], -395);


  // remove_if
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_if(thrust::device, device_ptr_A, device_ptr_A+10, pred);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("remove_if", host_ptr_R[0], -5);
  Report::check("remove_if", host_ptr_R[1], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_if(host_ptr_A, host_ptr_A+10, pred);
  Report::check("remove_if", host_ptr_A[0], -5);
  Report::check("remove_if", host_ptr_A[1], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, pred);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("remove_if", host_ptr_R[0], -5);
  Report::check("remove_if", host_ptr_R[1], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, pred);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("remove_if", host_ptr_R[0], -5);
  Report::check("remove_if", host_ptr_R[1], -395);

  // remove_copy_if
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_copy_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, pred);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("remove_copy_if", host_ptr_R[0], -5);
  Report::check("remove_copy_if", host_ptr_R[1], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_copy_if(host_ptr_A, host_ptr_A+10, host_ptr_R, pred);
  Report::check("remove_copy_if", host_ptr_R[0], -5);
  Report::check("remove_copy_if", host_ptr_R[1], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_copy_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, device_ptr_R, pred);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("remove_copy_if", host_ptr_R[0], -5);
  Report::check("remove_copy_if", host_ptr_R[1], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_copy_if(host_ptr_A, host_ptr_A+10, host_ptr_S, host_ptr_R, pred);
  Report::check("remove_copy_if", host_ptr_R[0], -5);
  Report::check("remove_copy_if", host_ptr_R[1], -395);

  // inclusive_scan_ptr
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::inclusive_scan(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("inclusive_scan_ptr", host_ptr_R[0], -5);
  Report::check("inclusive_scan_ptr", host_ptr_R[1], 3);
  Report::check("inclusive_scan_ptr", host_ptr_R[2], 399);
  Report::check("inclusive_scan_ptr", host_ptr_R[3], 4);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::inclusive_scan(host_ptr_A, host_ptr_A+10, host_ptr_R, thrust::plus<float>());
  Report::check("inclusive_scan_ptr", host_ptr_R[0], -5);
  Report::check("inclusive_scan_ptr", host_ptr_R[1], 3);
  Report::check("inclusive_scan_ptr", host_ptr_R[2], 399);
  Report::check("inclusive_scan_ptr", host_ptr_R[3], 4);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::inclusive_scan(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, thrust::plus<float>());
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("inclusive_scan_ptr", host_ptr_R[0], -5);
  Report::check("inclusive_scan_ptr", host_ptr_R[1], 3);
  Report::check("inclusive_scan_ptr", host_ptr_R[2], 399);
  Report::check("inclusive_scan_ptr", host_ptr_R[3], 4);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::inclusive_scan(host_ptr_A, host_ptr_A+10, host_ptr_R);
  Report::check("inclusive_scan_ptr", host_ptr_R[0], -5);
  Report::check("inclusive_scan_ptr", host_ptr_R[1], 3);
  Report::check("inclusive_scan_ptr", host_ptr_R[2], 399);
  Report::check("inclusive_scan_ptr", host_ptr_R[3], 4);

  // adjacent_difference
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("adjacent_difference_ptr", host_ptr_R[0], -5);
  Report::check("adjacent_difference_ptr", host_ptr_R[1], 13);
  Report::check("adjacent_difference_ptr", host_ptr_R[2], 388);
  Report::check("adjacent_difference_ptr", host_ptr_R[3], -791);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R, thrust::minus<float>());
  Report::check("adjacent_difference_ptr", host_ptr_R[0], -5);
  Report::check("adjacent_difference_ptr", host_ptr_R[1], 13);
  Report::check("adjacent_difference_ptr", host_ptr_R[2], 388);
  Report::check("adjacent_difference_ptr", host_ptr_R[3], -791);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::adjacent_difference(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_R, thrust::minus<float>());
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("adjacent_difference_ptr", host_ptr_R[0], -5);
  Report::check("adjacent_difference_ptr", host_ptr_R[1], 13);
  Report::check("adjacent_difference_ptr", host_ptr_R[2], 388);
  Report::check("adjacent_difference_ptr", host_ptr_R[3], -791);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::adjacent_difference(host_ptr_A, host_ptr_A+10, host_ptr_R);
  Report::check("adjacent_difference_ptr", host_ptr_R[0], -5);
  Report::check("adjacent_difference_ptr", host_ptr_R[1], 13);
  Report::check("adjacent_difference_ptr", host_ptr_R[2], 388);
  Report::check("adjacent_difference_ptr", host_ptr_R[3], -791);

  // gather
  int *host_ptr_M;
  int *device_ptr_M;
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  host_ptr_M = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  cudaMalloc(&device_ptr_M, 20 * sizeof(int));
  host_ptr_M[0]= 3;
  host_ptr_M[1]= 2;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  cudaMemcpy(device_ptr_M, host_ptr_M, 20*sizeof(int), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::gather(thrust::device, device_ptr_M, device_ptr_M + 4, device_ptr_A, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("gather", host_ptr_R[0], -395);
  Report::check("gather", host_ptr_R[1], 396);
  Report::check("gather", host_ptr_R[2], 8);
  Report::check("gather", host_ptr_R[3], -5);
  host_ptr_M[0]= 3;
  host_ptr_M[1]= 2;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  cudaMemcpy(device_ptr_M, host_ptr_M, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::gather(host_ptr_M, host_ptr_M + 4, host_ptr_A, host_ptr_R);
  Report::check("gather", host_ptr_R[0], -395);
  Report::check("gather", host_ptr_R[1], 396);
  Report::check("gather", host_ptr_R[2], 8);
  Report::check("gather", host_ptr_R[3], -5);

  // scatter
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  host_ptr_M = (int*)std::malloc(20 * sizeof(int));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  cudaMalloc(&device_ptr_M, 20 * sizeof(int));
  host_ptr_M[0]= 2;
  host_ptr_M[1]= 3;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  cudaMemcpy(device_ptr_M, host_ptr_M, 20*sizeof(int), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= 3;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::scatter(thrust::device, device_ptr_A, device_ptr_A + 4, device_ptr_M, device_ptr_R);
  cudaMemcpy(host_ptr_R, device_ptr_R, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("scatter", host_ptr_R[0], 3);
  Report::check("scatter", host_ptr_R[1], 396);
  Report::check("scatter", host_ptr_R[2], -5);
  Report::check("scatter", host_ptr_R[3], 8);
  host_ptr_M[0]= 2;
  host_ptr_M[1]= 3;
  host_ptr_M[2]= 1;
  host_ptr_M[3]= 0;
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= 3;
  thrust::scatter(host_ptr_A, host_ptr_A + 4, host_ptr_M, host_ptr_R);
  Report::check("scatter", host_ptr_R[0], 3);
  Report::check("scatter", host_ptr_R[1], 396);
  Report::check("scatter", host_ptr_R[2], -5);
  Report::check("scatter", host_ptr_R[3], 8);

  // unique_by_key_copy
  float *host_ptr_K;
  float *device_ptr_K;
  thrust::equal_to<float> pred2;
  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_K = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_K, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_K[0]= 1;
  host_ptr_K[1]= 2;
  host_ptr_K[2]= 2;
  host_ptr_K[3]= 1;
  cudaMemcpy(device_ptr_K, host_ptr_K, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::unique_by_key_copy(thrust::device, device_ptr_K, device_ptr_K + 4, device_ptr_A, device_ptr_R, device_ptr_S, pred2);
  cudaMemcpy(host_ptr_S, device_ptr_S, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("unique_by_key_copy", host_ptr_S[0], -5);
  Report::check("unique_by_key_copy", host_ptr_S[1], 8);
  Report::check("unique_by_key_copy", host_ptr_S[2], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::unique_by_key_copy(host_ptr_K, host_ptr_K+10, host_ptr_A, host_ptr_R, host_ptr_S, pred2);
  Report::check("unique_by_key_copy", host_ptr_S[0], -5);
  Report::check("unique_by_key_copy", host_ptr_S[1], 8);
  Report::check("unique_by_key_copy", host_ptr_S[2], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::unique_by_key_copy(thrust::device, device_ptr_K, device_ptr_K + 4, device_ptr_A, device_ptr_R, device_ptr_S);
  cudaMemcpy(host_ptr_S, device_ptr_S, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("unique_by_key_copy", host_ptr_S[0], -5);
  Report::check("unique_by_key_copy", host_ptr_S[1], 8);
  Report::check("unique_by_key_copy", host_ptr_S[2], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::unique_by_key_copy(host_ptr_K, host_ptr_K+10, host_ptr_A, host_ptr_R, host_ptr_S);
  Report::check("unique_by_key_copy", host_ptr_S[0], -5);
  Report::check("unique_by_key_copy", host_ptr_S[1], 8);
  Report::check("unique_by_key_copy", host_ptr_S[2], -395);
  return 0;
}
