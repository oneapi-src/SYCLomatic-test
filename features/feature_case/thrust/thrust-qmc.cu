// ====------ thrust-qmc.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <complex>
#include <iostream>

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <cuda_runtime.h>

#include "report.h"

// Temporary workaround for dpct::device_malloc assuming bytes.
// Once this is fixed the commented out version of this macro should be used
// #define MALLOC_PARAM(n, t) (n)
#define MALLOC_PARAM(n, t) ((n)*sizeof(t))

namespace qmc_cuda {
  void cuda_check(cudaError_t sucess, std::string message = "")
  {
    if(cudaSuccess != sucess) {
      std::cerr<<message <<std::endl;
      std::cerr<<" cudaGetErrorName: " <<cudaGetErrorName(sucess) <<std::endl;
      std::cerr<<" cudaGetErrorString: " <<cudaGetErrorString(sucess) <<std::endl;
      std::cerr.flush();
      throw std::runtime_error(" Error code returned by cuda. \n");
    }
  }
}

namespace kernels 
{

template<typename T>
__global__ void kernel_determinant_from_geqrf(int N, thrust::complex<T> *m, int lda, thrust::complex<T>* buff, thrust::complex<T> LogOverlapFactor, thrust::complex<T> *det) {

   __shared__ thrust::complex<T> tmp[256];
   int t = threadIdx.x;

//   printf("m[%d] = %f\n", t, m[t].real());

   tmp[t]=thrust::complex<T>(0.0);

   for(int ip=threadIdx.x; ip<N; ip+=blockDim.x)
   {
     if (m[ip*lda+ip].real() < 0)
       buff[ip]=thrust::complex<T>(-1.0);
     else
       buff[ip]=thrust::complex<T>(1.0);
     tmp[t] += thrust::log(buff[ip]*m[ip*lda+ip]);
   }
   __syncthreads();

//   printf("tmp[%d] = %f\n", t, tmp[t].real());

   // not optimal but ok for now
   if (threadIdx.x == 0) {
     int imax = (N > blockDim.x)?blockDim.x:N;
     for(int i=1; i<imax; i++)
       tmp[0] += tmp[i];
//     printf("tmp[0] = (%f, %f)\n", tmp[0].real(), tmp[0].imag());
     *det = thrust::exp(tmp[0]-LogOverlapFactor);
//     printf("*det = (%f, %f)\n", det->real(), det->imag());
   }
   __syncthreads();
}

std::complex<double> determinant_from_geqrf_gpu(int N, std::complex<double> *m, int lda, std::complex<double> *buff, std::complex<double> LogOverlapFactor)
{
  thrust::device_ptr<thrust::complex<double>> d_ptr = thrust::device_malloc<thrust::complex<double>>(MALLOC_PARAM(1, thrust::complex<double>));
  kernel_determinant_from_geqrf<<<1,256>>>(N,
                                    reinterpret_cast<thrust::complex<double> *>(m),lda,
                                    reinterpret_cast<thrust::complex<double> *>(buff), 
                                    static_cast<thrust::complex<double>>(LogOverlapFactor),
                                    thrust::raw_pointer_cast(d_ptr));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
  std::complex<double> res;
  qmc_cuda::cuda_check(cudaMemcpy(std::addressof(res),thrust::raw_pointer_cast(d_ptr),
                sizeof(std::complex<double>),cudaMemcpyDeviceToHost));
  thrust::device_free(d_ptr);
  return res;
}

}

void initArray(int N, std::complex<double> *m) {
  for (int i = 0; i < N; ++i) {
    m[i] = std::complex<double>((double)(i+1), (double)(i+1+N));
  }
}

bool checkFloat(std::complex<double> a, std::complex<double> b) {
  return abs(a.real() - b.real()) < 0.001 &&
         abs(a.imag() - b.imag()) < 0.001;
}

int main() {
  Report::start("QMCPack thrust usage");
  const int N = 256;
  std::complex<double>                     mH[N];
  thrust::device_ptr<std::complex<double>> mD   = thrust::device_malloc<std::complex<double>>(MALLOC_PARAM(N, std::complex<double>));
  thrust::device_ptr<std::complex<double>> buff = thrust::device_malloc<std::complex<double>>(MALLOC_PARAM(N, std::complex<double>));
  std::complex<double> LogOverlapFactor(1525.0);
  std::complex<double> d;
  initArray(N, mH);
  cudaMemcpy(thrust::raw_pointer_cast(mD), mH, N*sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  d = kernels::determinant_from_geqrf_gpu(N, thrust::raw_pointer_cast(mD), 0, thrust::raw_pointer_cast(buff), LogOverlapFactor);
  std::complex<double> expected(1507.418184, 197.083227);
  char buf[200];
  if (checkFloat(d, expected)) {
    sprintf(buf, "(%.2f, %.2f)", d.real(), d.imag());
    Report::pass(std::string("Correct result: ") + buf);
  }
  else {
    sprintf(buf, "result(%.2f, %.2f) != expected(%.2f, %.2f)", d.real(), d.imag(), expected.real(), expected.imag());
    Report::fail(std::string("Incorrect result: ") + buf);
  }
  return Report::finish();
}