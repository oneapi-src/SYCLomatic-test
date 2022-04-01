// ====------ texture_object.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// Simple transformation kernel
__global__ void transformKernel(float* output, cudaTextureObject_t texObj, int width, int height, float theta)
{
// Calculate normalized texture coordinates
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
float u = x / (float)width;
float v = y / (float)height;
// Transform coordinates
u -= 0.5f;
v -= 0.5f;
float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;
// Read from texture and write to global memory
output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

// Host code
int main()
{

float h_data[4][4];
// Allocate CUDA array in device memory
cudaChannelFormatDesc channelDesc =
cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
cudaArray* cuArray;
int width = 4;
int height = 4;
cudaMallocArray(&cuArray, &channelDesc, width, height);
// Copy to device memory some data located at address h_data
// in host memory
int size = width*height*sizeof(float);
cudaMemcpyToArray(cuArray, 0, 0, h_data, size,cudaMemcpyHostToDevice);
// Specify texture
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;
// Specify texture object parameters
struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeWrap;
texDesc.addressMode[1] = cudaAddressModeWrap;
texDesc.filterMode = cudaFilterModeLinear;
//texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 1;
// Create texture object
cudaTextureObject_t texObj = 0;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
// Allocate result of transformation in device memory
float* output;
cudaMalloc(&output, width * height * sizeof(float));
// Invoke kernel
dim3 dimBlock(height, width);
dim3 dimGrid(1, 1);
transformKernel<<<dimGrid, dimBlock>>>(output, texObj, width, height, 0);
// Destroy texture object
cudaDestroyTextureObject(texObj);
// Free device memory
cudaFreeArray(cuArray);
cudaFree(output);
return 0;
}
