// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<stdio.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include<functional>

using namespace std;

__global__ void mykernel(int *deviceData, int par){

    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    deviceData[tid]=deviceData[tid]+par;

}


void gFunc(cudaStream_t s, cudaError_t e){cout<<"gFunc"<<endl;}

//void addCallbackStream(std::function<void(cudaStream_t, cudaError_t)> func)
//{
//			auto funcPointer = new std::function<void(cudaStream_t, cudaError_t)>(func);
			//cudaSafeCall(cudaStreamAddCallback(m_associatedStream, &(Container<T>::cudaDeleteCallback), funcPointer, 0));
//}


int main(){
   
   std::function<void(cudaStream_t, cudaError_t)> f=gFunc;
   //f();


   return 0;

}
