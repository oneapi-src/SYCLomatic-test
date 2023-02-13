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
    int version, nranks = 2, rank = 0;
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

    MPI_Finalize();

    printf("TEST PASS\n");
}
