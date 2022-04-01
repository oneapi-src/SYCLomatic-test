// ====------ cudaMemcpyAsyncTest.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<stdio.h>
#include<cuda_runtime.h>
#include<thrust/transform.h>
#include<thrust/functional.h>
#include<thrust/execution_policy.h>
#include<thrust/tuple.h>
#include<cmath>
#include<algorithm>
#include<functional>




typedef cudaStream_t myStreamType;


 
void MemcpyAsyHostToDevice(int *des, int *src, int N,myStreamType myStream){
      cudaMemcpyAsync(des,src,N*sizeof(int),cudaMemcpyHostToDevice,myStream);    
}

void MemcpyAsyDeviceToHost(int *des, int *src, int N,myStreamType myStream){
      cudaMemcpyAsync(des,src,N*sizeof(int),cudaMemcpyDeviceToHost, myStream);    
}






__global__ void mykernel(int *deviceData, int par){

    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    deviceData[tid]=deviceData[tid]+par;

}



int main(){
   int N=10;
  
   int *h_array, *h_retri_array1,*h_retri_array2;
   cudaHostAlloc((void**)&h_array,N*sizeof(int),cudaHostAllocDefault);
   cudaHostAlloc((void**)&h_retri_array1,N*sizeof(int),cudaHostAllocDefault);
   cudaHostAlloc((void**)&h_retri_array2,N*sizeof(int),cudaHostAllocDefault);
   
   for(int i=0;i<N;i++)
        h_array[i]=i;
   
   int *d_array1,*d_array2;
   cudaMalloc((void**)&d_array1,N*sizeof(int));
   cudaMalloc((void**)&d_array2,N*sizeof(int));
  
   
  
    
  myStreamType stream1;
   
  
   cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
  


   MemcpyAsyHostToDevice(d_array1,h_array, N,stream1);
   MemcpyAsyHostToDevice(d_array2,h_array, N,cudaStreamDefault);


   mykernel<<<1,N,0,stream1>>>(d_array1,2);
   mykernel<<<1,N,0,cudaStreamDefault>>>(d_array2,4);


   MemcpyAsyDeviceToHost(h_retri_array1,d_array1, N,stream1);
   MemcpyAsyDeviceToHost(h_retri_array2,d_array2, N,cudaStreamDefault);
   cudaStreamSynchronize(stream1);
   cudaStreamSynchronize(cudaStreamDefault);

   printf("The values in h_retri_array1 is:\n");
   for(int i=0;i<N;i++)
       printf("%d\n",h_retri_array1[i]);
   printf("The values in h_retri_array2 is:\n");
   for(int i=0;i<N;i++)
       printf("%d\n",h_retri_array2[i]);


   cudaFreeHost(h_array);
   cudaFreeHost(h_retri_array1);
   cudaFreeHost(h_retri_array2);
   cudaFree(d_array1);
   cudaFree(d_array2);
   cudaStreamDestroy(stream1);



   return 0;

}
