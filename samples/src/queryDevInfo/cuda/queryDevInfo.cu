// ====------ queryDevInfo.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<ctime>

// Get the P2P info between device in the same host.
int InfoQuery() {
    int dCount = 0;
    cudaGetDeviceCount(&dCount);
    for (size_t dev1 = 0; dev1 < dCount; dev1++) {
        for (size_t dev2 = 0; dev2 < dCount; dev2++) {
            if (dev1 == dev2) continue;
            int perfRank = 0;
            int Access = 0;
            int nativeAtomic = 0;
            int arrayAcess = 0;
            cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, dev1, dev2);
            cudaDeviceGetP2PAttribute(&Access, cudaDevP2PAttrAccessSupported, dev1, dev2);
            cudaDeviceGetP2PAttribute(&nativeAtomic, cudaDevP2PAttrNativeAtomicSupported, dev1, dev2);
            cudaDeviceGetP2PAttribute(&arrayAcess, cudaDevP2PAttrCudaArrayAccessSupported, dev1, dev2);

            if (Access) {
                std::cout << "The Src device: " << dev1 << " <-> the Dst device: " << dev2 << std::endl;
                std::cout << " Atomic support: " << (nativeAtomic ? "yes" : "no") << std::endl;
            }
        }
    }

    for (size_t dev = 0; dev < dCount; dev++) {
        int clockRate = 0;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
        std::cout << "The max clock rate of dev: " << dev << " , value is " << clockRate << std::endl;
    }
    return 0;
}


int main() {
    InfoQuery();
    return 0;
}
