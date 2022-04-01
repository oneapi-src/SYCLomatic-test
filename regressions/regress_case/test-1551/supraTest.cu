// ====------ supraTest.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<stdio.h>
#include<cuda_runtime.h>



__global__ void mykernel(int *deviceData, int par){

    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    deviceData[tid]=deviceData[tid]+par;

}


int main(){
   int N=10;
   int *h_array=(int*)malloc(N*sizeof(int));
   int *h_retri_array1=(int*)malloc(N*sizeof(int));
   
   
   for(int i=0;i<N;i++)
        h_array[i]=i;
   
   int *d_array1;
   cudaMalloc((void**)&d_array1,N*sizeof(int));
   cudaMemcpy(d_array1,h_array,N*sizeof(int),cudaMemcpyHostToDevice);
   dim3 blockSize(256,1,1);
   dim3 gridSize(
				static_cast<unsigned int>((N + blockSize.x - 1) / blockSize.x),
				static_cast<unsigned int>((N+ blockSize.y - 1) / blockSize.y));
   int l=blockSize.x;

   mykernel<<<1,N>>>(d_array1,2);
   cudaMemcpy(h_retri_array1,d_array1,N*sizeof(int),cudaMemcpyDeviceToHost);
  

   printf("The values in h_retri_array1 is:\n");
   for(int i=0;i<N;i++)
       printf("%d\n",h_retri_array1[i]);
   


   free(h_array);
   free(h_retri_array1);
   //free(h_retri_array2);
   cudaFree(d_array1);
   //cudaFree(d_array2);
   //cudaStreamDestroy(stream1);
   //cudaStreamDestroy(stream2);
   //cudaStreamDestroy(p_stream[0]);
   //cudaStreamDestroy(p_stream[1]);
   //delete [] p_stream;

   return 0;

}
