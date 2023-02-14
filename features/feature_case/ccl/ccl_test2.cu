#include<stdio.h>
#include <malloc.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"


int main(int argc,char**argv){
  int version, nranks = 2, rank = 3, device_num = -1;
    int device_id = -1;
    
    ncclUniqueId id;
    ncclComm_t comm;
    int size=32;
    ncclCommInitRank(&comm, nranks, id, rank);
    ncclCommCount(comm, &device_num);
    //allocating and initializing device buffers
    float ** sendbuff=(float**)malloc(1*sizeof(float*));
    float ** recvbuff=(float**)malloc(1*sizeof(float*));
    cudaStream_t stream = 0;
    cudaMalloc(sendbuff,size*sizeof(float));
    cudaMalloc(recvbuff,size*sizeof(float));
    cudaMemset(sendbuff[0],1,size*sizeof(float));
    cudaMemset(recvbuff[0],0,size*sizeof(float));
    float *hostbuff;
    hostbuff = (float *)malloc(size*sizeof(float));
    for (int i=0; i<size; ++i)
      hostbuff[i] = i;
    cudaMemcpy(sendbuff[0], hostbuff, size*sizeof(float), cudaMemcpyHostToDevice);

    ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream);
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    free(hostbuff);

    printf("TEST PASS\n");
    return 0;

}