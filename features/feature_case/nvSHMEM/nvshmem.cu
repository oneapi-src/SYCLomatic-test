// ===--------------- nvshmem.cu --------------- *- CUDA -* ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <iostream>
#include <cassert>

#define N 32

__global__ void set_data(int *shared_data, int mype) {
  size_t i = threadIdx.x;

  shared_data[i] = static_cast<int>(mype * 2);
}

__global__ void kernel_putmem_signal_nbi(int *shared_data, uint64_t *signal, int val, int target_pe) {
  nvshmem_putmem_signal_nbi(shared_data, shared_data, N * sizeof(int), signal, val, NVSHMEM_SIGNAL_SET, target_pe);
}

__global__ void kernel_signal_wait_until(uint64_t *signal, uint64_t val) {
  nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, val);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  nvshmemx_init_attr_t attr;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();

  std::cout << "ISHMEM initialized with " << npes << " PEs." << std::endl;
  std::cout << "My PE: " << mype << std::endl;

  int *shared_data = (int *)nvshmem_malloc(N * sizeof(int));
  assert(shared_data != nullptr && "nvshmem_malloc failed");

  uint64_t *signal_addr = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
  assert(signal_addr != nullptr && "nvshmem_malloc for signal failed");

  set_data<<<N, N>>>(shared_data, mype);

  const int target_pe = 1;

  // copy data from PE 0 to PE 1
  if (mype == 0) {
    nvshmem_putmem_nbi(shared_data, shared_data, N * sizeof(int), target_pe);
  }

  int recv_shared_data[N] = {1};

  // Retrieve the data on PE 1
  if (mype == 1) {
    cudaMemcpy((void *)recv_shared_data, (void *)shared_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
      if (recv_shared_data[i] != 0) {
        std::cerr << "[" << mype << "] Data verification 1 failed at index " << i << ": " << recv_shared_data[i] << "\n";
        std::exit(1);
      }
    }
  }

  // Get the pointer to data on PE 1
  int *remote_data = (int *)nvshmem_ptr(shared_data, 1);
  cudaMemcpy((void *)recv_shared_data, (void *)remote_data, N * sizeof(int), cudaMemcpyDeviceToHost);

  if (mype == 0) {
    for (int i = 0; i < N; i++) {
      if (recv_shared_data[i] != 0) {
        std::cerr << "[" << mype << "] Data verification 2 failed at index " << i << ": " << recv_shared_data[i] << "\n";
        std::exit(1);
      }
    }
  }
  std::cout << "putmem_nbi & shmem_ptr: Data verification successful" << std::endl;

  // Reset data on all PEs
  set_data<<<N, N>>>(shared_data, mype);

  // Allocate & set signal memory
  uint64_t *signal = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
  assert(signal != nullptr && "nvshmem_malloc failed");
  
  uint64_t h_signal = 0;
  cudaMemcpy((void *)signal, (void *)&h_signal, sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Copy data from PE 0 to PE 1 and signal completion
  // nvshmem_barrier_all();
  kernel_putmem_signal_nbi<<<1, 1>>>(shared_data, signal, 1, 1);

  // Check whether signal value & data updated in PE 1
  if (mype == 1) {
    kernel_signal_wait_until<<<1, 1>>>(signal, 1);

    cudaMemcpy((void *)recv_shared_data, (void *)shared_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
      if (recv_shared_data[i] != 0) {
        std::cerr << "[" << mype << "] Data verification 3 failed at index " << i << ": " << recv_shared_data[i] << "\n";
        std::exit(1);
      }
    }
  }
  std::cout << "putmem_signal_nbi & signal_wait_until: Data verification successful" << std::endl;

  // Set and update signal value in PE 1
  nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, 1);
  nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_ADD, 1);

  // Check whether signal value updated in PE 1
  if (mype == 1) {
    kernel_signal_wait_until<<<1, 1>>>(signal, 2);

    cudaMemcpy((void *)recv_shared_data, (void *)signal, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    if (recv_shared_data[0] != 2) {
      std::cerr << "[" << mype << "] Signal verification failed: " << recv_shared_data[0] << "\n";
      std::exit(1);
    }
  }
  std::cout << "signal_set/add: Data verification successful" << std::endl;

  nvshmem_free(shared_data);
  nvshmem_free(signal);

  nvshmem_finalize();
  MPI_Finalize();

  return 0;
}
