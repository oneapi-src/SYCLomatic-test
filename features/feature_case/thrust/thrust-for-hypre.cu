// ====------ thrust-for-hypre.cu---------- *- CUDA -* ----===////
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
#include "report.h"
// for cuda 12.0
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

struct greater_than_zero
{
  __host__ __device__
  bool operator()(int x) const
  {
    return x > 0;
  }
  typedef int argument_type;
};

template<class T>
void print(T t, int size){
  std::cout<<std::endl;
  for(int i = 0; i < size; i++){
    std::cout<<t[i]<<" ";
  }
  std::cout<<std::endl;
}

template<class T1, class T2>
void foo2(T1 policy, T2 vec){
  thrust::device_vector<int> R(4);
  thrust::inclusive_scan(policy, vec.begin(), vec.end(), R.begin(), thrust::multiplies<int>());
  Report::check("inclusive_scan", R[0], -5);
  Report::check("inclusive_scan", R[1], -40);
  Report::check("inclusive_scan", R[2], -15840);
  Report::check("inclusive_scan", R[3], 6256800);
}

void foo_host(){
  thrust::device_vector<int> A(4);
  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;
  thrust::device_vector<int> S(4);
  S[0] = -1;
  S[1] =  0;
  S[2] = -1;
  S[3] =  1;
  thrust::device_vector<int> R(4);

  std::vector<int> B(4);
  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;
  std::vector<int> S2(4);
  S2[0] = -1;
  S2[1] =  0;
  S2[2] = -1;
  S2[3] =  1;
  std::vector<int> R2(4);

  greater_than_zero pred;

  thrust::equal_to<int>();

  thrust::less<int>();

  thrust::not1(pred);

  thrust::replace_if(thrust::device, A.begin(), A.end(), pred, 0);
  Report::check("replace_if", A[0], -5);
  Report::check("replace_if", A[1], 0);
  Report::check("replace_if", A[2], 0);
  Report::check("replace_if", A[3], 0);
  thrust::replace_if(A.begin(), A.end(), pred, 0);
  Report::check("replace_if", A[0], -5);
  Report::check("replace_if", A[1], 0);
  Report::check("replace_if", A[2], 0);
  Report::check("replace_if", A[3], 0);
  thrust::replace_if(thrust::device, A.begin(), A.end(), S.begin(), pred, 0);
  Report::check("replace_if", A[0], -5);
  Report::check("replace_if", A[1], 0);
  Report::check("replace_if", A[2], 0);
  Report::check("replace_if", A[3], 0);
  thrust::replace_if(A.begin(), A.end(), S.begin(), pred, 0);
  Report::check("replace_if", A[0], -5);
  Report::check("replace_if", A[1], 0);
  Report::check("replace_if", A[2], 0);
  Report::check("replace_if", A[3], 0);
  thrust::replace_if(thrust::seq, B.begin(), B.end(), pred, 0);
  Report::check("replace_if", B[0], -5);
  Report::check("replace_if", B[1], 0);
  Report::check("replace_if", B[2], 0);
  Report::check("replace_if", B[3], 0);
  thrust::replace_if(B.begin(), B.end(), pred, 0);
  Report::check("replace_if", B[0], -5);
  Report::check("replace_if", B[1], 0);
  Report::check("replace_if", B[2], 0);
  Report::check("replace_if", B[3], 0);
  thrust::replace_if(thrust::seq, B.begin(), B.end(), S2.begin(), pred, 0);
  Report::check("replace_if", B[0], -5);
  Report::check("replace_if", B[1], 0);
  Report::check("replace_if", B[2], 0);
  Report::check("replace_if", B[3], 0);
  thrust::replace_if(B.begin(), B.end(), S2.begin(), pred, 0);
  Report::check("replace_if", B[0], -5);
  Report::check("replace_if", B[1], 0);
  Report::check("replace_if", B[2], 0);
  Report::check("replace_if", B[3], 0);

  // corner case of replace_if
  {
    thrust::host_vector<long long> vA(1);
    thrust::host_vector<long long> vB(1);

    vA[0] = 0xd61f6d2a3364f9c0;
    vB[0] = 0;
    thrust::replace_if(thrust::host,vA.begin(),vA.end(),vB.begin(),pred,0);
    Report::check("replace_if with long long vector", vA[0], 0xd61f6d2a3364f9c0);
  }

  // numerically the same as the replace_if test, but use replace_copy_if

  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;

  S[0] = -1;
  S[1] =  0;
  S[2] = -1;
  S[3] =  1;

  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;

  S2[0] = -1;
  S2[1] =  0;
  S2[2] = -1;
  S2[3] =  1;

  thrust::replace_copy_if(thrust::device, A.begin(), A.end(), A.begin(), pred, 0);
  Report::check("replace_copy_if", A[0], -5);
  Report::check("replace_copy_if", A[1], 0);
  Report::check("replace_copy_if", A[2], 0);
  Report::check("replace_copy_if", A[3], 0);
  thrust::replace_copy_if(A.begin(), A.end(), A.begin(), pred, 0);
  Report::check("replace_copy_if", A[0], -5);
  Report::check("replace_copy_if", A[1], 0);
  Report::check("replace_copy_if", A[2], 0);
  Report::check("replace_copy_if", A[3], 0);
  thrust::replace_copy_if(thrust::device, A.begin(), A.end(), S.begin(), A.begin(), pred, 0);
  Report::check("replace_copy_if", A[0], -5);
  Report::check("replace_copy_if", A[1], 0);
  Report::check("replace_copy_if", A[2], 0);
  Report::check("replace_copy_if", A[3], 0);
  thrust::replace_copy_if(A.begin(), A.end(), S.begin(), A.begin(), pred, 0);
  Report::check("replace_copy_if", A[0], -5);
  Report::check("replace_copy_if", A[1], 0);
  Report::check("replace_copy_if", A[2], 0);
  Report::check("replace_copy_if", A[3], 0);
  thrust::replace_copy_if(thrust::seq, B.begin(), B.end(), B.begin(), pred, 0);
  Report::check("replace_copy_if", B[0], -5);
  Report::check("replace_copy_if", B[1], 0);
  Report::check("replace_copy_if", B[2], 0);
  Report::check("replace_copy_if", B[3], 0);
  thrust::replace_copy_if(B.begin(), B.end(), B.begin(), pred, 0);
  Report::check("replace_copy_if", B[0], -5);
  Report::check("replace_copy_if", B[1], 0);
  Report::check("replace_copy_if", B[2], 0);
  Report::check("replace_copy_if", B[3], 0);
  thrust::replace_copy_if(thrust::seq, B.begin(), B.end(), S2.begin(), B.begin(), pred, 0);
  Report::check("replace_copy_if", B[0], -5);
  Report::check("replace_copy_if", B[1], 0);
  Report::check("replace_copy_if", B[2], 0);
  Report::check("replace_copy_if", B[3], 0);
  thrust::replace_copy_if(B.begin(), B.end(), S2.begin(), B.begin(), pred, 0);
  Report::check("replace_copy_if", B[0], -5);
  Report::check("replace_copy_if", B[1], 0);
  Report::check("replace_copy_if", B[2], 0);
  Report::check("replace_copy_if", B[3], 0);

  // corner case of replace_copy_if
  {
    thrust::host_vector<long long> vA(1);
    thrust::host_vector<long long> vB(1);
    thrust::host_vector<long long> vC(1);

    vA[0] = 0xfffffffec19cb1d0LL;
    vB[0] = 0;
    vC[0] = 0;
    thrust::replace_copy_if(vA.begin(),vA.end(),vB.begin(),vC.begin(),pred,0);
    Report::check("replace_copy_if with long long vector", vC[0], 0xfffffffec19cb1d0LL);
  }

  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;
  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;
  // [Todo] Currently the migraion of the last 2 cases of remove_if are incorrect since
  // dpct::remove_if(oneapi::dpl::execution::seq, B.begin(), B.end(), S2.begin(), pred)
  // will encouter compile fail in oneDPL. Will fix after ONEDPL-271 is fixed.
  thrust::remove_if(thrust::device, A.begin(), A.end(), pred);
  Report::check("remove_if", A[0], -5);
  thrust::remove_if(A.begin(), A.end(), pred);
  Report::check("remove_if", A[0], -5);
  thrust::remove_if(thrust::device, A.begin(), A.end(), S.begin(), pred);
  Report::check("remove_if", A[0], -5);
  thrust::remove_if(A.begin(), A.end(), S.begin(), pred);
  Report::check("remove_if", A[0], -5);
  thrust::remove_if(thrust::seq, B.begin(), B.end(), pred);
  Report::check("remove_if", B[0], -5);
  thrust::remove_if(B.begin(), B.end(), pred);
  Report::check("remove_if", B[0], -5);
  thrust::remove_if(thrust::seq, B.begin(), B.end(), S2.begin(), pred);
  Report::check("remove_if", B[0], -5);
  thrust::remove_if(B.begin(), B.end(), S2.begin(), pred);
  Report::check("remove_if", B[0], -5);

  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;
  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;
  thrust::remove_copy_if(thrust::device, A.begin(), A.end(), R.begin(), pred);
  Report::check("remove_copy_if", R[0], -5);
  thrust::remove_copy_if(A.begin(), A.end(), R.begin(), pred);
  Report::check("remove_copy_if", R[0], -5);
  thrust::remove_copy_if(thrust::device, A.begin(), A.end(), S.begin(), R.begin(), pred);
  Report::check("remove_copy_if", R[0], -5);
  thrust::remove_copy_if(A.begin(), A.end(), S.begin(), R.begin(), pred);
  Report::check("remove_copy_if", R[0], -5);
  thrust::remove_copy_if(thrust::seq, B.begin(), B.end(), R2.begin(), pred);
  Report::check("remove_copy_if", R2[0], -5);
  thrust::remove_copy_if(B.begin(), B.end(), R2.begin(), pred);
  Report::check("remove_copy_if", R2[0], -5);
  thrust::remove_copy_if(thrust::seq, B.begin(), B.end(), S2.begin(), R2.begin(), pred);
  Report::check("remove_copy_if", R2[0], -5);
  thrust::remove_copy_if(B.begin(), B.end(), S2.begin(), R2.begin(), pred);
  Report::check("remove_copy_if", R2[0], -5);

  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;
  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;
  bool Bool_R = 0;
  Bool_R = thrust::any_of(A.begin(), A.end(), pred);
  Report::check("any_of", Bool_R, 1);
  Bool_R = thrust::any_of(B.begin(), B.end(), pred);
  Report::check("any_of", Bool_R, 1);
  Bool_R = thrust::any_of(thrust::device, A.begin(), A.end(), pred);
  Report::check("any_of", Bool_R, 1);
  Bool_R = thrust::any_of(thrust::seq, B.begin(), B.end(), pred);
  Report::check("any_of", Bool_R, 1);

  A[0] = -5;
  A[1] = 3;
  A[2] = 0;
  A[3] = 4;
  B[0] = -5;
  B[1] = 3;
  B[2] = 0;
  B[3] = 4;
  thrust::replace(A.begin(), A.end(), 0, 399);
  Report::check("replace", A[0], -5);
  Report::check("replace", A[1], 3);
  Report::check("replace", A[2], 399);
  Report::check("replace", A[3], 4);
  thrust::replace(B.begin(), B.end(), 0, 399);
  Report::check("replace", B[0], -5);
  Report::check("replace", B[1], 3);
  Report::check("replace", B[2], 399);
  Report::check("replace", B[3], 4);
  thrust::replace(thrust::device, A.begin(), A.end(), 0, 399);
  Report::check("replace", A[0], -5);
  Report::check("replace", A[1], 3);
  Report::check("replace", A[2], 399);
  Report::check("replace", A[3], 4);
  thrust::replace(thrust::seq, B.begin(), B.end(), 0, 399);
  Report::check("replace", B[0], -5);
  Report::check("replace", B[1], 3);
  Report::check("replace", B[2], 399);
  Report::check("replace", B[3], 4);

  A[0] = -5;
  A[1] = 3;
  A[2] = 399;
  A[3] = 4;
  B[0] = -5;
  B[1] = 3;
  B[2] = 399;
  B[3] = 4;
  #define TM thrust::multiplies<int>()
  thrust::adjacent_difference(A.begin(), A.end(), R.begin(), TM);
  Report::check("adjacent_difference", R[0], -5);
  Report::check("adjacent_difference", R[1], -15);
  Report::check("adjacent_difference", R[2], 1197);
  Report::check("adjacent_difference", R[3], 1596);
  thrust::adjacent_difference(B.begin(), B.end(), R2.begin(), thrust::minus<int>());
  Report::check("adjacent_difference", R2[0], -5);
  Report::check("adjacent_difference", R2[1], 8);
  Report::check("adjacent_difference", R2[2], 396);
  Report::check("adjacent_difference", R2[3], -395);
  thrust::adjacent_difference(thrust::device, A.begin(), A.end(), R.begin(), thrust::minus<int>());
  Report::check("adjacent_difference", R[0], -5);
  Report::check("adjacent_difference", R[1], 8);
  Report::check("adjacent_difference", R[2], 396);
  Report::check("adjacent_difference", R[3], -395);
  thrust::adjacent_difference(thrust::seq, B.begin(), B.end(), R2.begin(), thrust::minus<int>());
  Report::check("adjacent_difference", R2[0], -5);
  Report::check("adjacent_difference", R2[1], 8);
  Report::check("adjacent_difference", R2[2], 396);
  Report::check("adjacent_difference", R2[3], -395);
  thrust::adjacent_difference(A.begin(), A.end(), R.begin());
  Report::check("adjacent_difference", R[0], -5);
  Report::check("adjacent_difference", R[1], 8);
  Report::check("adjacent_difference", R[2], 396);
  Report::check("adjacent_difference", R[3], -395);
  thrust::adjacent_difference(B.begin(), B.end(), R2.begin());
  Report::check("adjacent_difference", R2[0], -5);
  Report::check("adjacent_difference", R2[1], 8);
  Report::check("adjacent_difference", R2[2], 396);
  Report::check("adjacent_difference", R2[3], -395);
  thrust::adjacent_difference(thrust::device, A.begin(), A.end(), R.begin());
  Report::check("adjacent_difference", R[0], -5);
  Report::check("adjacent_difference", R[1], 8);
  Report::check("adjacent_difference", R[2], 396);
  Report::check("adjacent_difference", R[3], -395);
  thrust::adjacent_difference(thrust::seq, B.begin(), B.end(), R2.begin());
  Report::check("adjacent_difference", R2[0], -5);
  Report::check("adjacent_difference", R2[1], 8);
  Report::check("adjacent_difference", R2[2], 396);
  Report::check("adjacent_difference", R2[3], -395);

  A[0] = -5;
  A[1] = 8;
  A[2] = 396;
  A[3] = -395;
  B[0] = -5;
  B[1] = 8;
  B[2] = 396;
  B[3] = -395;
  thrust::inclusive_scan(A.begin(), A.end(), R.begin(), TM);
  Report::check("inclusive_scan", R[0], -5);
  Report::check("inclusive_scan", R[1], -40);
  Report::check("inclusive_scan", R[2], -15840);
  Report::check("inclusive_scan", R[3], 6256800);
  thrust::inclusive_scan(B.begin(), B.end(), R2.begin(), thrust::multiplies<int>());
  Report::check("inclusive_scan", R2[0], -5);
  Report::check("inclusive_scan", R2[1], -40);
  Report::check("inclusive_scan", R2[2], -15840);
  Report::check("inclusive_scan", R2[3], 6256800);
  thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin(), thrust::multiplies<int>());
  Report::check("inclusive_scan", R[0], -5);
  Report::check("inclusive_scan", R[1], -40);
  Report::check("inclusive_scan", R[2], -15840);
  Report::check("inclusive_scan", R[3], 6256800);
  thrust::inclusive_scan(thrust::seq, B.begin(), B.end(), R2.begin(), thrust::multiplies<int>());
  Report::check("inclusive_scan", R2[0], -5);
  Report::check("inclusive_scan", R2[1], -40);
  Report::check("inclusive_scan", R2[2], -15840);
  Report::check("inclusive_scan", R2[3], 6256800);
  thrust::inclusive_scan(A.begin(), A.end(), R.begin());
  Report::check("inclusive_scan", R[0], -5);
  Report::check("inclusive_scan", R[1], 3);
  Report::check("inclusive_scan", R[2], 399);
  Report::check("inclusive_scan", R[3], 4);
  thrust::inclusive_scan(B.begin(), B.end(), R2.begin());
  Report::check("inclusive_scan", R2[0], -5);
  Report::check("inclusive_scan", R2[1], 3);
  Report::check("inclusive_scan", R2[2], 399);
  Report::check("inclusive_scan", R2[3], 4);
  thrust::inclusive_scan(thrust::device, A.begin(), A.end(), R.begin());
  Report::check("inclusive_scan", R[0], -5);
  Report::check("inclusive_scan", R[1], 3);
  Report::check("inclusive_scan", R[2], 399);
  Report::check("inclusive_scan", R[3], 4);
  thrust::inclusive_scan(thrust::seq, B.begin(), B.end(), R2.begin());
  Report::check("inclusive_scan", R2[0], -5);
  Report::check("inclusive_scan", R2[1], 3);
  Report::check("inclusive_scan", R2[2], 399);
  Report::check("inclusive_scan", R2[3], 4);
  foo2(thrust::device ,A);




  float *host_ptr_A;
  float *host_ptr_R;
  float *host_ptr_S;
  float *device_ptr_A;
  float *device_ptr_S;
  float *device_ptr_R;


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
  thrust::inclusive_scan(thrust::seq, host_ptr_A, host_ptr_A+10, host_ptr_R);
  Report::check("inclusive_scan_ptr", host_ptr_R[0], -5);
  Report::check("inclusive_scan_ptr", host_ptr_R[1], 3);
  Report::check("inclusive_scan_ptr", host_ptr_R[2], 399);
  Report::check("inclusive_scan_ptr", host_ptr_R[3], 4);

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
  Report::check("adjacent_difference", host_ptr_R[0], -5);
  Report::check("adjacent_difference", host_ptr_R[1], 13);
  Report::check("adjacent_difference", host_ptr_R[2], 388);
  Report::check("adjacent_difference", host_ptr_R[3], -791);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 396;
  host_ptr_A[3]= -395;
  thrust::adjacent_difference(thrust::seq, host_ptr_A, host_ptr_A+10, host_ptr_R);
  Report::check("adjacent_difference", host_ptr_R[0], -5);
  Report::check("adjacent_difference", host_ptr_R[1], 13);
  Report::check("adjacent_difference", host_ptr_R[2], 388);
  Report::check("adjacent_difference", host_ptr_R[3], -791);

  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 0;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::replace(thrust::device, device_ptr_A, device_ptr_A + 10, 0, 399);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  Report::check("replace", host_ptr_R[0], -5);
  Report::check("replace", host_ptr_R[1], 8);
  Report::check("replace", host_ptr_R[2], 399);
  Report::check("replace", host_ptr_R[3], -395);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 0;
  host_ptr_A[3]= -395;
  thrust::replace(host_ptr_A, host_ptr_A + 10, 0, 399);
  Report::check("replace", host_ptr_A[0], -5);
  Report::check("replace", host_ptr_A[1], 8);
  Report::check("replace", host_ptr_A[2], 399);
  Report::check("replace", host_ptr_A[3], -395);

  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
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
  thrust::replace_if(thrust::seq, host_ptr_A, host_ptr_A+10, pred, 0);
  Report::check("replace_if", host_ptr_A[0], -5);
  Report::check("replace_if", host_ptr_A[1], 0);
  Report::check("replace_if", host_ptr_A[2], 0);
  Report::check("replace_if", host_ptr_A[3], -395);

  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, pred);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_if(thrust::seq, host_ptr_A, host_ptr_A+10, pred);
  Report::check("remove_if", host_ptr_A[0], -5);
  Report::check("remove_if", host_ptr_A[1], -395);

  host_ptr_A = (float*)std::malloc(20 * sizeof(float));
  host_ptr_R = (float*)std::malloc(20 * sizeof(float));
  host_ptr_S = (float*)std::malloc(20 * sizeof(float));
  cudaMalloc(&device_ptr_A, 20 * sizeof(float));
  cudaMalloc(&device_ptr_S, 20 * sizeof(float));
  cudaMalloc(&device_ptr_R, 20 * sizeof(float));
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  cudaMemcpy(device_ptr_A, host_ptr_A, 20*sizeof(float), cudaMemcpyHostToDevice);
  host_ptr_S[0]= -1;
  host_ptr_S[1]= 5;
  host_ptr_S[2]= 5;
  host_ptr_S[3]= -395;
  cudaMemcpy(device_ptr_S, host_ptr_S, 20*sizeof(float), cudaMemcpyHostToDevice);
  thrust::remove_copy_if(thrust::device, device_ptr_A, device_ptr_A+10, device_ptr_S, device_ptr_R, pred);
  cudaMemcpy(host_ptr_R, device_ptr_A, 20*sizeof(float), cudaMemcpyDeviceToHost);
  host_ptr_A[0]= -5;
  host_ptr_A[1]= 8;
  host_ptr_A[2]= 50;
  host_ptr_A[3]= -395;
  thrust::remove_copy_if(thrust::seq, host_ptr_A, host_ptr_A+10, host_ptr_R, pred);
  Report::check("remove_if", host_ptr_R[0], -5);
  Report::check("remove_if", host_ptr_R[1], -395);
}


int main(){
  foo_host();
  return Report::finish();;
}

