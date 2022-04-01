// ====------ test.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <helper_cuda.h>
//#include <helper_math.h>
//#include "math_vector.h"

#define MAX_INSTANCES 2
cudaResourceDesc resDescInput[MAX_INSTANCES];
cudaTextureDesc texDescInput[MAX_INSTANCES];
cudaArray *d_Input[MAX_INSTANCES] = { NULL };
cudaTextureObject_t tex_Input[MAX_INSTANCES] = { NULL };



void
createTestTexture(int instance, unsigned char *d_In, int rSize, int pSize, int pPitch)
{
	// create texture object
	memset(&resDescInput[instance], 0, sizeof(resDescInput[instance]));
	resDescInput[instance].resType = cudaResourceTypePitch2D;
	resDescInput[instance].res.pitch2D.devPtr = d_In;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	resDescInput[instance].res.pitch2D.desc = channelDesc;
	resDescInput[instance].res.pitch2D.height = pSize;
	resDescInput[instance].res.pitch2D.width = rSize;
	resDescInput[instance].res.pitch2D.pitchInBytes = pPitch;

	memset(&texDescInput[instance], 0, sizeof(texDescInput[instance]));
	//texDescInput[instance].readMode = cudaReadModeNormalizedFloat;
	texDescInput[instance].filterMode = cudaFilterModeLinear;
	texDescInput[instance].normalizedCoords = false;
	texDescInput[instance].addressMode[0] = cudaAddressModeBorder;
	texDescInput[instance].addressMode[1] = cudaAddressModeBorder;

	if (tex_Input[instance] != NULL)
	{
		cudaDestroyTextureObject(tex_Input[instance]);
		tex_Input[instance] = NULL;
	}
	cudaCreateTextureObject(&tex_Input[instance],
		&resDescInput[instance],
		&texDescInput[instance], NULL);
}


void
createTestTextureAlternative(int instance, unsigned char *d_In, int rSize, int pSize, int pPitch)
{
	// create texture object
	memset(&resDescInput[instance], 0, sizeof(resDescInput[instance]));
	resDescInput[instance].resType = cudaResourceTypeArray;
	resDescInput[instance].res.array.array = d_Input[instance];

	memset(&texDescInput[instance], 0, sizeof(texDescInput[instance]));
	//texDescInput[instance].readMode = cudaReadModeNormalizedFloat;
	texDescInput[instance].filterMode = cudaFilterModeLinear;
	texDescInput[instance].normalizedCoords = false;
	texDescInput[instance].addressMode[0] = cudaAddressModeBorder;
	texDescInput[instance].addressMode[1] = cudaAddressModeBorder;

	if (tex_Input[instance] != NULL)
	{
		cudaDestroyTextureObject(tex_Input[instance]);
		tex_Input[instance] = NULL;
	}
	cudaCreateTextureObject(&tex_Input[instance],
		&resDescInput[instance],
		&texDescInput[instance], NULL);
}




__global__ void
test_Kernel(cudaTextureObject_t tex_inArg, float *d_out, int yPitchOutInFloat)
{
	// x and y are coordinates of the output 2D array
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;
	int i = y * yPitchOutInFloat + x;

    float rTextureCoord = x;
	float pTextureCoord = y;
    float V = tex2D<float>(tex_inArg, rTextureCoord + 0.5f, pTextureCoord + 0.5f);

    d_out[i]=V;
}


void test(float *d_Out, int rSize, int pSize, int pPitch)
{
	int numThreadsPerBlock = 128;
	int blocks = (rSize + numThreadsPerBlock - 1) / numThreadsPerBlock;
	dim3 blockSz(numThreadsPerBlock);
	dim3 gridSz(blocks, pSize);
	test_Kernel <<<gridSz, blockSz >>>(tex_Input[0], d_Out, pPitch/sizeof(float));
}
int main() {
  return 0;
}
