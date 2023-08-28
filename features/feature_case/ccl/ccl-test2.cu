// ====------ ccl-test2.cu-------------------- *- CUDA -* --////////////--===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <mpi.h>
#include <nccl.h>
#include "cuda_runtime.h"

int main() {
    int version, nranks = 2, rank = 1, device_num = -1;
    int device_id = -1;
    int rank_g=0;
    cudaStream_t stream=0;
    size_t count = 10*1024;
    float *sendbuff, *recvbuff,*hostbuff = (float *)malloc(count * sizeof(float));
    for(int i =1;i<count+1;++i) *(hostbuff+i-1)=i;

    cudaMalloc(&sendbuff, count * sizeof(float));
    cudaMemcpy(sendbuff, hostbuff, sizeof(float) * count, cudaMemcpyHostToDevice);
    
    ncclUniqueId id;
    ncclComm_t comm;

    ncclGetVersion(&version);

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
      ncclGetUniqueId(&id);

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaSetDevice(0);

    ncclCommInitRank(&comm, nranks, id, rank);

    ncclCommUserRank(comm, &rank_g);

    ncclBroadcast(sendbuff, sendbuff, count, ncclFloat, rank_g, comm, stream);
    cudaStreamSynchronize(stream);
    ncclBcast(sendbuff, count, ncclFloat, rank_g, comm, stream);
    cudaStreamSynchronize(stream);
    ncclCommDestroy(comm);
    MPI_Finalize();
    cudaFree(sendbuff);
    free(hostbuff);

    printf("TEST PASS\n");
}
