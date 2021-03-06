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
    std::string Path, FunctionName, Data;
    FunctionName = "foo";
    cuModuleLoad(&M, Path.c_str());
    cuModuleLoadData(&M, Data.c_str());
    cuModuleGetFunction(&F, M, FunctionName.c_str());
    float *param[2] = {0};
    cudaMalloc(&param[0], sizeof(float));
    cudaMalloc(&param[1], sizeof(float));
    cuLaunchKernel(F, 1, 1, 1, 1, 1, 1, 10, 0, (void**)param, nullptr);
    CUtexref tex;
    cuModuleGetTexRef(&tex, M, "tex");
    cuModuleUnload(M);
    cudaDeviceSynchronize();
    cudaFree(param[0]);
    cudaFree(param[1]);
    return 0;
}
