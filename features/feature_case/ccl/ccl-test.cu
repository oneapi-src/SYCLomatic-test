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
    
    int device_id = -1;
    ncclCommCuDevice(comm, &device_id);
    std::cout<<"device_id: "<<device_id<<std::endl;

    MPI_Finalize();
    if(device_num != 2) {//hard code
      printf("TEST failed for ncclCommCount\n");
      return 1;
    } 
      
    printf("TEST PASS\n");
    return 0;
}
