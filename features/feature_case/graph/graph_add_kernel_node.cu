#include <cuda_runtime.h>
#include <iostream>

// Kernel 1: Increment each element of the array
__global__ void incrementKernel(float *a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += 1.0f;
    }
}

// Kernel 2: Multiply each element of the array by a scalar
__global__ void multiplyKernel(float *a, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] *= scalar;
    }
}

int main() {
    const int size = 10;
    const int bytes = size * sizeof(float);

    // Allocate and initialize host memory
    float *h_a = new float[size];
    for (int i = 0; i < size; i++) {
        h_a[i] = static_cast<float>(i + 1);
    }

    // Allocate device memory
    float *d_a;
    cudaMalloc(&d_a, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Begin stream capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Launch the increment kernel in the stream
    incrementKernel<<<1, size, 0, stream>>>(d_a, size);

    // End stream capture and create a graph
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);

    // Add two multiplyKernel nodes to the graph
    dim3 threadsPerBlock(10);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Define kernel parameters for the first multiplyKernel node
    float scalar1 = 2.0f;
    void *kernelArgs1[] = {&d_a, &scalar1, (void*)&size};
    cudaKernelNodeParams kernelNodeParams1 = {};
    kernelNodeParams1.func = (void *)multiplyKernel;
    kernelNodeParams1.gridDim = blocksPerGrid;
    kernelNodeParams1.blockDim = threadsPerBlock;
    kernelNodeParams1.sharedMemBytes = 0;
    kernelNodeParams1.kernelParams = kernelArgs1;

    cudaGraphNode_t kernelNode1;
    cudaGraphAddKernelNode(&kernelNode1, graph, nullptr, 0, &kernelNodeParams1);

    // Define kernel parameters for the second multiplyKernel node
    // float scalar2 = 3.0f;
    // void *kernelArgs2[] = {&d_a, &scalar2, (void*)&size};
    // cudaKernelNodeParams kernelNodeParams2 = {};
    // kernelNodeParams2.func = (void *)multiplyKernel;
    // kernelNodeParams2.gridDim = blocksPerGrid;
    // kernelNodeParams2.blockDim = threadsPerBlock;
    // kernelNodeParams2.sharedMemBytes = 0;
    // kernelNodeParams2.kernelParams = kernelArgs2;

    // cudaGraphNode_t kernelNode2;
    // cudaGraphAddKernelNode(&kernelNode2, graph, &kernelNode1, 1, &kernelNodeParams2);

    // Instantiate and launch the graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(graphExec, stream);

    // Wait for the graph to complete
    cudaStreamSynchronize(stream);

    // Copy the result back to the host
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < size; i++) {
        std::cout << "h_a[" << i << "] = " << h_a[i] << std::endl;
    }

    // Clean up
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    delete[] h_a;

    return 0;
}