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
#include <memory>



using namespace std;
using namespace thrust::placeholders;
struct thrustLogcompress : public thrust::unary_function<float,float>{};

template<typename T>
class Container 
	{
	public:
		

		Container(cudaStream_t associatedStream, size_t numel)
		{
			
			m_numel = numel;
		
			m_associatedStream = associatedStream;

			
		};

        private:
           size_t m_numel;
           cudaStream_t m_associatedStream;
	};



__global__ void mykernel(int *deviceData, int par){

    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    deviceData[tid]=deviceData[tid]+par;

}



int main(){
   int N=10;
  
   int *h_array=(int *)malloc(N*sizeof(int));
   int *h_retri_array1=(int *)malloc(N*sizeof(int));
   
   
   
   
   for(int i=0;i<N;i++)
        h_array[i]=i;
   
   int *d_array1;
   cudaMalloc((void**)&d_array1,N*sizeof(int));

   std::shared_ptr<Container<uint8_t> > m_mask;
  m_mask= make_shared<Container<uint8_t> >(cudaStreamPerThread, N*sizeof(int));

   cudaMemcpyAsync(d_array1, h_array, 10 * sizeof(int), cudaMemcpyHostToDevice,cudaStreamDefault);

   mykernel<<<1,N,0,cudaStreamDefault>>>(d_array1,4);
     

   
   cudaMemcpyAsync(h_retri_array1,d_array1, N*sizeof(int),cudaMemcpyDeviceToHost,cudaStreamDefault);
   
   cudaStreamSynchronize(cudaStreamDefault);

   printf("The values in h_retri_array1 is:\n");
   for(int i=0;i<N;i++)
       printf("%d\n",h_retri_array1[i]);
 


   free(h_array);
   free(h_retri_array1);
   
   cudaFree(d_array1);
   
  



   return 0;

}
