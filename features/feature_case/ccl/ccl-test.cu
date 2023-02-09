// ====------ ccl-test.cu-------------------- *- CUDA -* --////////////--===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <mpi.h>
#include <nccl.h>

int main() {
    int version, nranks = 2, rank = 3, device_num = -1;
    int device_id = -1;
    cudaStream_t stream = 0;
    size_t count = 4;
    float *sendbuff, *recvbuff,*hostbuff = (float *)malloc(count * sizeof(float));
    for(int i =1;i<5;++i) *(hostbuff+i-1)=i;
    cudaMalloc(&sendbuff, count * sizeof(float));
    cudaMalloc(&recvbuff, count * sizeof(float));
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

    ncclCommInitRank(&comm, nranks, id, rank);

    ncclCommCount(comm, &device_num);

    ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
    
    MPI_Finalize();
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    free(hostbuff);
    printf("TEST PASS\n");
    return 0;
}
