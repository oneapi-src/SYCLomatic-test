// ====------ module-main.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <string>
int main(){
    CUmodule M;
    CUfunction F;
#ifdef _WIN32
    std::string Path{"./module-kernel.dll"};
#else
    std::string Path{"./module-kernel.so"};
#endif
    std::string FunctionName{"foo"}, Data;
    FunctionName = "foo";
    cuModuleLoad(&M, Path.c_str());
    cuModuleGetFunction(&F, M, FunctionName.c_str());
    float **param[2] = {0};
    float *p0, *p1;
    cudaMalloc(&p0, sizeof(float));
    cudaMalloc(&p1, sizeof(float));
    param[0] = &p0;
    param[1] = &p1;
    cuLaunchKernel(F, 1, 1, 1, 1, 1, 1, 10, 0, (void**)param, nullptr);
    CUtexref tex;
    cuModuleGetTexRef(&tex, M, "tex");
    cuModuleUnload(M);
    cudaDeviceSynchronize();
    cudaFree(param[0]);
    cudaFree(param[1]);
    return 0;
}
