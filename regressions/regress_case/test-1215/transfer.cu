// ====------ transfer.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_CHECK(x) \
	do { \
		cudaError_t last_err = (x); \
		if (last_err != cudaSuccess) { \
			fprintf(stderr, "%s:%u: CUDA error: %s\n", __FILE__, __LINE__, \
					cudaGetErrorString(last_err)); \
			exit(1); \
		} \
	} while (false)

#define CUDA_CALL(x) CUDA_ERROR_CHECK(x)

int main(void)
{  
	const size_t buf_count = 1024;
	const size_t buf_size = buf_count * sizeof(double);

	double *buf = (double*) calloc(buf_size, buf_size);
	double *buf_dev;

	CUDA_CALL(cudaMalloc((void**)&buf_dev, buf_size)); 
	CUDA_CALL(cudaMemcpy(buf_dev, buf, buf_size, cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	CUDA_CALL(cudaMemcpy(buf, buf_dev, buf_size, cudaMemcpyDeviceToHost));
	
	CUDA_CALL(cudaFree(buf_dev));
	free(buf);
	
	return EXIT_SUCCESS;
}
