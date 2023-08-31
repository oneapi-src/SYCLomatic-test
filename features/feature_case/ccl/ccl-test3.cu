// ====------ ccl-test3.cu---------------------------------- *- CUDA -* ----===//
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

int main()
{
  int version, nranks = 2, rank = 1, device_num = -1;
  int device_id = -1;
  int rank_g = 0;
  cudaStream_t stream = 0;
  size_t count = 1024;
  float *sendbuff, *recvbuff, *hostbuff = (float *)malloc(count * sizeof(float));
  for (int i = 0; i < count; ++i)
    *(hostbuff + i) = i + 1;

  cudaMalloc(&sendbuff, count * sizeof(float));
  cudaMalloc(&recvbuff, count * sizeof(float));
  cudaMemcpy(sendbuff, hostbuff, sizeof(float) * count, cudaMemcpyHostToDevice);
  for (int i = 0; i < count; ++i)
    *(hostbuff + i) = 0;
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
  ncclSend(sendbuff, count, ncclFloat, rank_g, comm, stream);
  ncclRecv(recvbuff, count, ncclFloat, rank_g, comm, stream);

  cudaStreamSynchronize(stream);
  cudaMemcpy(hostbuff, recvbuff, sizeof(float) * count, cudaMemcpyDeviceToHost);
  ncclCommDestroy(comm);
  MPI_Finalize();
  for (int i = 0; i < count; ++i)
  {
    if (*(hostbuff + i) != i + 1)
    {
      return 1;
    }
  }
  cudaFree(sendbuff);
  cudaFree(recvbuff);
  free(hostbuff);

  printf("TEST PASS\n");
  return 0;
}
