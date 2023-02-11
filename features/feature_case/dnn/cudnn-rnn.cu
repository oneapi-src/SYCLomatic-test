// ====------ cudnn-rnn.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>

template<cudnnDataType_t T>
struct dt_trait{
    typedef void type;
};
template<>
struct dt_trait<CUDNN_DATA_FLOAT>{
    typedef float type;
};

template<>
struct dt_trait<CUDNN_DATA_INT32>{
    typedef int type;
};
template<>
struct dt_trait<CUDNN_DATA_HALF>{
    typedef float type;
};

template<typename T>
void check(std::vector<T> &expect, T *actual, int num, float precision) {
  for(int i = 0; i < num; i++){
      if(std::abs(expect[i] - actual[i]) > precision) {
          std::cout << "test failed" << std::endl;
          std::cout << "expect:" << expect[i] << std::endl;
          std::cout << "actual:" << actual[i] << std::endl;
          exit(-1);
      }
  }
}

template<typename T = float>
void initData(T *data, T init, int size, bool inc = false){
  for(int i = 0; i < size; i++) {
      data[i] = init + (inc ? 1.f : 0.f);
  }
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test1() {

    int hidenSize = 2;
    int layerSize = 3;
    int inputSize = 2;
    int projectSize = 2;
    int batchSize = 3;
    int maxSeqLength = 4;
    int dir = 1;

    int hDim[3] = {dir * layerSize, batchSize, projectSize};
    int hStride[3] = {hDim[1] * hDim[2], hDim[2], 1};

    int cDim[3] = {layerSize * dir, batchSize, hidenSize};
    int cStride[3] = {cDim[2] * cDim[1], cDim[2], 1};
    int xDim[3] = {maxSeqLength, batchSize, inputSize};
    int yDim[3] = {maxSeqLength, batchSize, dir * projectSize};

    int h_size = hDim[0] * hDim[1] * hDim[2];
    int c_size = cDim[0] * cDim[1] * cDim[2];
    int x_size = xDim[0] * xDim[1] * xDim[2];
    int y_size = yDim[0] * yDim[1] * yDim[2];

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnRNNDescriptor_t rnnDesc;
    cudnnRNNDataDescriptor_t xDesc;
    cudnnRNNDataDescriptor_t yDesc;
    cudnnTensorDescriptor_t hDesc;
    cudnnTensorDescriptor_t cDesc;

    cudnnCreateRNNDescriptor(&rnnDesc);
    cudnnCreateRNNDataDescriptor(&xDesc);
    cudnnCreateRNNDataDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&hDesc);
    cudnnCreateTensorDescriptor(&cDesc);

    size_t spacesize, statesize;
    void* reservespace, *state;

    cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, hDim, hStride);
    cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, cDim, cStride);

    cudnnSetRNNDescriptor_v8(rnnDesc,
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_RNN_RELU,
        CUDNN_RNN_SINGLE_INP_BIAS,
        dir == 1 ? CUDNN_UNIDIRECTIONAL : CUDNN_BIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_FLOAT,
        CUDNN_DEFAULT_MATH,
        inputSize,  // inputSize
        hidenSize,  // hiddenSize
        projectSize,  // projSize
        layerSize,  // numLayers
        //dropoutDesc,
        NULL,
        CUDNN_RNN_PADDED_IO_ENABLED
    );

    int seqLenArray[3];
    seqLenArray[0] = maxSeqLength;
    seqLenArray[1] = maxSeqLength;
    seqLenArray[2] = maxSeqLength;

    cudnnSetRNNDataDescriptor(xDesc, 
        CUDNN_DATA_FLOAT, 
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        xDim[0], // maxSeqLength
        xDim[1],  // batchSize
        xDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );

    cudnnSetRNNDataDescriptor(yDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        yDim[0], // maxSeqLength
        yDim[1],  // batchSize
        yDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );
    size_t weightsSpaceSize, workSpaceSize, reserveSpaceSize;
    auto ss = cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightsSpaceSize);
    cudnnGetRNNTempSpaceSizes(handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING,
        xDesc, 
        &workSpaceSize,
        &reserveSpaceSize
    );

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;
    cudaMalloc(&xData, sizeof(float) * x_size);
    cudaMalloc(&yData, sizeof(float)  * y_size);
    cudaMalloc(&hxData, sizeof(float) * h_size);
    cudaMalloc(&hyData, sizeof(float) * h_size);
    cudaMalloc(&cxData, sizeof(float) * c_size);
    cudaMalloc(&cyData, sizeof(float) * c_size);
    cudaMalloc(&weightsData, weightsSpaceSize);
    cudaMalloc(&workSpaceData, workSpaceSize);
    cudaMalloc(&reserveSpaceData, reserveSpaceSize);

    cudaMalloc(&dxData, sizeof(float) * x_size);
    cudaMalloc(&dyData, sizeof(float)  * y_size);
    cudaMalloc(&dhxData, sizeof(float) * h_size);
    cudaMalloc(&dhyData, sizeof(float) * h_size);
    cudaMalloc(&dcxData, sizeof(float) * c_size);
    cudaMalloc(&dcyData, sizeof(float) * c_size);
    cudaMalloc(&dweightsData, weightsSpaceSize);

    float *host_xData, *host_yData, *host_hxData, *host_hyData, *host_cxData, *host_cyData, *host_weightsData, *host_workSpaceData, *host_reserveSpaceData;
    float *host_dxData, *host_dyData, *host_dhxData, *host_dhyData, *host_dcxData, *host_dcyData, *host_dweightsData;
    host_xData = (float *)malloc(sizeof(float) * x_size);
    host_yData = (float *)malloc(sizeof(float) * y_size);
    host_hxData = (float *)malloc(sizeof(float) * h_size);
    host_hyData = (float *)malloc(sizeof(float) * h_size);
    host_cxData = (float *)malloc(sizeof(float) * c_size);
    host_cyData = (float *)malloc(sizeof(float) * c_size);
    host_weightsData = (float *)malloc(weightsSpaceSize);
    host_workSpaceData = (float *)malloc(workSpaceSize);
    host_reserveSpaceData = (float *)malloc(reserveSpaceSize);

    host_dxData = (float *)malloc(sizeof(float) * x_size);
    host_dyData = (float *)malloc(sizeof(float) * y_size);
    host_dhxData = (float *)malloc(sizeof(float) * h_size);
    host_dhyData = (float *)malloc(sizeof(float) * h_size);
    host_dcxData = (float *)malloc(sizeof(float) * c_size);
    host_dcyData = (float *)malloc(sizeof(float) * c_size);
    host_dweightsData = (float *)malloc(weightsSpaceSize);

    initData(host_xData, 1.0f, x_size);
    initData(host_yData, 1.0f, y_size);
    initData(host_hxData, 1.0f, h_size);
    initData(host_hyData, 1.0f, h_size);
    initData(host_cxData, 1.0f, c_size);
    initData(host_cyData, 1.0f, c_size);
    initData(host_weightsData, 1.0f, weightsSpaceSize  / sizeof(float));
    initData(host_workSpaceData, 1.f, workSpaceSize / sizeof(float));
    initData(host_reserveSpaceData, 1.f, reserveSpaceSize / sizeof(float));

    initData(host_dxData, 1.0f, x_size);
    initData(host_dyData, 1.0f, y_size);
    initData(host_dhxData, 1.0f, h_size);
    initData(host_dhyData, 1.0f, h_size);
    initData(host_dcxData, 1.0f, c_size);
    initData(host_dcyData, 1.0f, c_size);
    initData(host_dweightsData, 0.f, weightsSpaceSize  / sizeof(float));

    cudaMemcpy(xData, host_xData, sizeof(float) * x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(yData, host_yData, sizeof(float) * y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(hxData, host_hxData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(hyData, host_hyData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cxData, host_cxData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cyData, host_cyData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(weightsData, host_weightsData, weightsSpaceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(workSpaceData, host_workSpaceData, workSpaceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(reserveSpaceData, host_reserveSpaceData, reserveSpaceSize, cudaMemcpyHostToDevice);

    cudaMemcpy(dxData, host_dxData, sizeof(float) * x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dyData, host_dyData, sizeof(float) * y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhxData, host_dhxData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhyData, host_dhyData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcxData, host_dcxData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcyData, host_dcyData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dweightsData, host_dweightsData, weightsSpaceSize, cudaMemcpyHostToDevice);

    int *seqlenarray;
    cudaMalloc(&seqlenarray, sizeof(int) * 3);
    cudaMemcpy(seqlenarray, seqLenArray, sizeof(int) * 3, cudaMemcpyHostToDevice);

    auto e = cudnnRNNForward(
        handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING, 
        seqlenarray, 
        xDesc, 
        xData, 
        yDesc, 
        yData, 
        hDesc, 
        hxData, 
        hyData, 
        cDesc, 
        cxData,
        cyData,
        weightsSpaceSize, 
        weightsData, 
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    cudaMemcpy(host_yData, yData, sizeof(float) * y_size, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        29, 29,
        29, 29,
        29, 29,
        165, 165,
        165, 165,
        165, 165,
        661, 661,
        661, 661,
        661, 661,
        2229, 2229,
        2229, 2229,
        2229, 2229,
      };

    check(expect, host_yData, expect.size(), 1e-3);
    cudnnDestroy(handle);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test2() {

    int hidenSize = 2;
    int layerSize = 3;
    int inputSize = 2;
    int projectSize = 2;
    int batchSize = 3;
    int maxSeqLength = 4;
    int dir = 2;

    int hDim[3] = {dir * layerSize, batchSize, projectSize};
    int hStride[3] = {hDim[1] * hDim[2], hDim[2], 1};

    int cDim[3] = {layerSize * dir, batchSize, hidenSize};
    int cStride[3] = {cDim[2] * cDim[1], cDim[2], 1};
    int xDim[3] = {maxSeqLength, batchSize, inputSize};
    int yDim[3] = {maxSeqLength, batchSize, dir * projectSize};

    int h_size = hDim[0] * hDim[1] * hDim[2];
    int c_size = cDim[0] * cDim[1] * cDim[2];
    int x_size = xDim[0] * xDim[1] * xDim[2];
    int y_size = yDim[0] * yDim[1] * yDim[2];

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnRNNDescriptor_t rnnDesc;
    cudnnRNNDataDescriptor_t xDesc;
    cudnnRNNDataDescriptor_t yDesc;
    cudnnTensorDescriptor_t hDesc;
    cudnnTensorDescriptor_t cDesc;

    cudnnCreateRNNDescriptor(&rnnDesc);
    cudnnCreateRNNDataDescriptor(&xDesc);
    cudnnCreateRNNDataDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&hDesc);
    cudnnCreateTensorDescriptor(&cDesc);

    size_t spacesize, statesize;
    void* reservespace, *state;

    cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, hDim, hStride);
    cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, cDim, cStride);

    cudnnSetRNNDescriptor_v8(rnnDesc,
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_RNN_RELU,
        CUDNN_RNN_SINGLE_INP_BIAS,
        dir == 1 ? CUDNN_UNIDIRECTIONAL : CUDNN_BIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_FLOAT,
        CUDNN_DEFAULT_MATH,
        inputSize,  // inputSize
        hidenSize,  // hiddenSize
        projectSize,  // projSize
        layerSize,  // numLayers
        //dropoutDesc,
        NULL,
        CUDNN_RNN_PADDED_IO_ENABLED
    );

    int seqLenArray[3];
    seqLenArray[0] = maxSeqLength;
    seqLenArray[1] = maxSeqLength;
    seqLenArray[2] = maxSeqLength;

    cudnnSetRNNDataDescriptor(xDesc, 
        CUDNN_DATA_FLOAT, 
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        xDim[0], // maxSeqLength
        xDim[1],  // batchSize
        xDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );

    cudnnSetRNNDataDescriptor(yDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        yDim[0], // maxSeqLength
        yDim[1],  // batchSize
        yDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );
    size_t weightsSpaceSize, workSpaceSize, reserveSpaceSize;
    auto ss = cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightsSpaceSize);
    cudnnGetRNNTempSpaceSizes(handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING,
        xDesc, 
        &workSpaceSize,
        &reserveSpaceSize
    );

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;
    cudaMalloc(&xData, sizeof(float) * x_size);
    cudaMalloc(&yData, sizeof(float)  * y_size);
    cudaMalloc(&hxData, sizeof(float) * h_size);
    cudaMalloc(&hyData, sizeof(float) * h_size);
    cudaMalloc(&cxData, sizeof(float) * c_size);
    cudaMalloc(&cyData, sizeof(float) * c_size);
    cudaMalloc(&weightsData, weightsSpaceSize);
    cudaMalloc(&workSpaceData, workSpaceSize);
    cudaMalloc(&reserveSpaceData, reserveSpaceSize);

    cudaMalloc(&dxData, sizeof(float) * x_size);
    cudaMalloc(&dyData, sizeof(float)  * y_size);
    cudaMalloc(&dhxData, sizeof(float) * h_size);
    cudaMalloc(&dhyData, sizeof(float) * h_size);
    cudaMalloc(&dcxData, sizeof(float) * c_size);
    cudaMalloc(&dcyData, sizeof(float) * c_size);
    cudaMalloc(&dweightsData, weightsSpaceSize);

    float *host_xData, *host_yData, *host_hxData, *host_hyData, *host_cxData, *host_cyData, *host_weightsData, *host_workSpaceData, *host_reserveSpaceData;
    float *host_dxData, *host_dyData, *host_dhxData, *host_dhyData, *host_dcxData, *host_dcyData, *host_dweightsData;
    host_xData = (float *)malloc(sizeof(float) * x_size);
    host_yData = (float *)malloc(sizeof(float) * y_size);
    host_hxData = (float *)malloc(sizeof(float) * h_size);
    host_hyData = (float *)malloc(sizeof(float) * h_size);
    host_cxData = (float *)malloc(sizeof(float) * c_size);
    host_cyData = (float *)malloc(sizeof(float) * c_size);
    host_weightsData = (float *)malloc(weightsSpaceSize);
    host_workSpaceData = (float *)malloc(workSpaceSize);
    host_reserveSpaceData = (float *)malloc(reserveSpaceSize);

    host_dxData = (float *)malloc(sizeof(float) * x_size);
    host_dyData = (float *)malloc(sizeof(float) * y_size);
    host_dhxData = (float *)malloc(sizeof(float) * h_size);
    host_dhyData = (float *)malloc(sizeof(float) * h_size);
    host_dcxData = (float *)malloc(sizeof(float) * c_size);
    host_dcyData = (float *)malloc(sizeof(float) * c_size);
    host_dweightsData = (float *)malloc(weightsSpaceSize);

    initData(host_xData, 1.0f, x_size);
    initData(host_yData, 1.0f, y_size);
    initData(host_hxData, 1.0f, h_size);
    initData(host_hyData, 1.0f, h_size);
    initData(host_cxData, 1.0f, c_size);
    initData(host_cyData, 1.0f, c_size);
    initData(host_weightsData, 1.0f, weightsSpaceSize  / sizeof(float));
    initData(host_workSpaceData, 1.f, workSpaceSize / sizeof(float));
    initData(host_reserveSpaceData, 1.f, reserveSpaceSize / sizeof(float));

    initData(host_dxData, 1.0f, x_size);
    initData(host_dyData, 1.0f, y_size);
    initData(host_dhxData, 1.0f, h_size);
    initData(host_dhyData, 1.0f, h_size);
    initData(host_dcxData, 1.0f, c_size);
    initData(host_dcyData, 1.0f, c_size);
    initData(host_dweightsData, 0.f, weightsSpaceSize  / sizeof(float));

    cudaMemcpy(xData, host_xData, sizeof(float) * x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(yData, host_yData, sizeof(float) * y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(hxData, host_hxData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(hyData, host_hyData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cxData, host_cxData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cyData, host_cyData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(weightsData, host_weightsData, weightsSpaceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(workSpaceData, host_workSpaceData, workSpaceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(reserveSpaceData, host_reserveSpaceData, reserveSpaceSize, cudaMemcpyHostToDevice);

    cudaMemcpy(dxData, host_dxData, sizeof(float) * x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dyData, host_dyData, sizeof(float) * y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhxData, host_dhxData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhyData, host_dhyData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcxData, host_dcxData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcyData, host_dcyData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dweightsData, host_dweightsData, weightsSpaceSize, cudaMemcpyHostToDevice);

    int *seqlenarray;
    cudaMalloc(&seqlenarray, sizeof(int) * 3);
    cudaMemcpy(seqlenarray, seqLenArray, sizeof(int) * 3, cudaMemcpyHostToDevice);

    auto e = cudnnRNNForward(
        handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING, 
        seqlenarray, 
        xDesc, 
        xData, 
        yDesc, 
        yData, 
        hDesc, 
        hxData, 
        hyData, 
        cDesc, 
        cxData,
        cyData,
        weightsSpaceSize, 
        weightsData, 
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    cudaMemcpy(host_yData, yData, sizeof(float) * y_size, cudaMemcpyDeviceToHost);

    std::vector<float> expect = {
        3719, 3719, 47275, 47275,
        3719, 3719, 47275, 47275,
        3719, 3719, 47275, 47275,

        9739, 9739, 21779, 21779,
        9739, 9739, 21779, 21779,
        9739, 9739, 21779, 21779,
        
        21779, 21779, 9739, 9739,
        21779, 21779, 9739, 9739,
        21779, 21779, 9739, 9739,
        
        47275, 47275, 3719, 3719,
        47275, 47275, 3719, 3719,
        47275, 47275, 3719, 3719,
      };

    check(expect, host_yData, expect.size(), 1e-3);
    cudnnDestroy(handle);
}

template<cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void test3() {

    int hidenSize = 2;
    int layerSize = 3;
    int inputSize = 2;
    int projectSize = 2;
    int batchSize = 3;
    int maxSeqLength = 4;
    int dir = 1;

    int hDim[3] = {dir * layerSize, batchSize, projectSize};
    int hStride[3] = {hDim[1] * hDim[2], hDim[2], 1};

    int cDim[3] = {layerSize * dir, batchSize, hidenSize};
    int cStride[3] = {cDim[2] * cDim[1], cDim[2], 1};
    int xDim[3] = {maxSeqLength, batchSize, inputSize};
    int yDim[3] = {maxSeqLength, batchSize, dir * projectSize};

    int h_size = hDim[0] * hDim[1] * hDim[2];
    int c_size = cDim[0] * cDim[1] * cDim[2];
    int x_size = xDim[0] * xDim[1] * xDim[2];
    int y_size = yDim[0] * yDim[1] * yDim[2];

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnRNNDescriptor_t rnnDesc;
    cudnnRNNDataDescriptor_t xDesc;
    cudnnRNNDataDescriptor_t yDesc;
    cudnnTensorDescriptor_t hDesc;
    cudnnTensorDescriptor_t cDesc;

    cudnnCreateRNNDescriptor(&rnnDesc);
    cudnnCreateRNNDataDescriptor(&xDesc);
    cudnnCreateRNNDataDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&hDesc);
    cudnnCreateTensorDescriptor(&cDesc);

    size_t spacesize, statesize;
    void* reservespace, *state;

    cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, hDim, hStride);
    cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, cDim, cStride);

    cudnnSetRNNDescriptor_v8(rnnDesc,
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_RNN_RELU,
        CUDNN_RNN_SINGLE_INP_BIAS,
        dir == 1 ? CUDNN_UNIDIRECTIONAL : CUDNN_BIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_FLOAT,
        CUDNN_DEFAULT_MATH,
        inputSize,  // inputSize
        hidenSize,  // hiddenSize
        projectSize,  // projSize
        layerSize,  // numLayers
        //dropoutDesc,
        NULL,
        CUDNN_RNN_PADDED_IO_ENABLED
    );

    int seqLenArray[3];
    seqLenArray[0] = maxSeqLength;
    seqLenArray[1] = maxSeqLength;
    seqLenArray[2] = maxSeqLength;

    cudnnSetRNNDataDescriptor(xDesc, 
        CUDNN_DATA_FLOAT, 
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        xDim[0], // maxSeqLength
        xDim[1],  // batchSize
        xDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );

    cudnnSetRNNDataDescriptor(yDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
        yDim[0], // maxSeqLength
        yDim[1],  // batchSize
        yDim[2],  // vectorSize
        seqLenArray, // seqLengthArray
        NULL
    );
    size_t weightsSpaceSize, workSpaceSize, reserveSpaceSize;
    auto ss = cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightsSpaceSize);
    cudnnGetRNNTempSpaceSizes(handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING,
        xDesc, 
        &workSpaceSize,
        &reserveSpaceSize
    );

    float *xData, *yData, *hxData, *hyData, *cxData, *cyData, *weightsData, *workSpaceData, *reserveSpaceData;
    float *dxData, *dyData, *dhxData, *dhyData, *dcxData, *dcyData, *dweightsData;
    cudaMalloc(&xData, sizeof(float) * x_size);
    cudaMalloc(&yData, sizeof(float)  * y_size);
    cudaMalloc(&hxData, sizeof(float) * h_size);
    cudaMalloc(&hyData, sizeof(float) * h_size);
    cudaMalloc(&cxData, sizeof(float) * c_size);
    cudaMalloc(&cyData, sizeof(float) * c_size);
    cudaMalloc(&weightsData, weightsSpaceSize);
    cudaMalloc(&workSpaceData, workSpaceSize);
    cudaMalloc(&reserveSpaceData, reserveSpaceSize);

    cudaMalloc(&dxData, sizeof(float) * x_size);
    cudaMalloc(&dyData, sizeof(float)  * y_size);
    cudaMalloc(&dhxData, sizeof(float) * h_size);
    cudaMalloc(&dhyData, sizeof(float) * h_size);
    cudaMalloc(&dcxData, sizeof(float) * c_size);
    cudaMalloc(&dcyData, sizeof(float) * c_size);
    cudaMalloc(&dweightsData, weightsSpaceSize);

    float *host_xData, *host_yData, *host_hxData, *host_hyData, *host_cxData, *host_cyData, *host_weightsData, *host_workSpaceData, *host_reserveSpaceData;
    float *host_dxData, *host_dyData, *host_dhxData, *host_dhyData, *host_dcxData, *host_dcyData, *host_dweightsData;
    host_xData = (float *)malloc(sizeof(float) * x_size);
    host_yData = (float *)malloc(sizeof(float) * y_size);
    host_hxData = (float *)malloc(sizeof(float) * h_size);
    host_hyData = (float *)malloc(sizeof(float) * h_size);
    host_cxData = (float *)malloc(sizeof(float) * c_size);
    host_cyData = (float *)malloc(sizeof(float) * c_size);
    host_weightsData = (float *)malloc(weightsSpaceSize);
    host_workSpaceData = (float *)malloc(workSpaceSize);
    host_reserveSpaceData = (float *)malloc(reserveSpaceSize);

    host_dxData = (float *)malloc(sizeof(float) * x_size);
    host_dyData = (float *)malloc(sizeof(float) * y_size);
    host_dhxData = (float *)malloc(sizeof(float) * h_size);
    host_dhyData = (float *)malloc(sizeof(float) * h_size);
    host_dcxData = (float *)malloc(sizeof(float) * c_size);
    host_dcyData = (float *)malloc(sizeof(float) * c_size);
    host_dweightsData = (float *)malloc(weightsSpaceSize);

    initData(host_xData, 1.0f, x_size);
    initData(host_yData, 1.0f, y_size);
    initData(host_hxData, 1.0f, h_size);
    initData(host_hyData, 1.0f, h_size);
    initData(host_cxData, 1.0f, c_size);
    initData(host_cyData, 1.0f, c_size);
    initData(host_weightsData, 1.0f, weightsSpaceSize  / sizeof(float));
    initData(host_workSpaceData, 1.f, workSpaceSize / sizeof(float));
    initData(host_reserveSpaceData, 1.f, reserveSpaceSize / sizeof(float));

    initData(host_dxData, 1.0f, x_size);
    initData(host_dyData, 1.0f, y_size);
    initData(host_dhxData, 1.0f, h_size);
    initData(host_dhyData, 1.0f, h_size);
    initData(host_dcxData, 1.0f, c_size);
    initData(host_dcyData, 1.0f, c_size);
    initData(host_dweightsData, 0.f, weightsSpaceSize  / sizeof(float));

    cudaMemcpy(xData, host_xData, sizeof(float) * x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(yData, host_yData, sizeof(float) * y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(hxData, host_hxData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(hyData, host_hyData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cxData, host_cxData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cyData, host_cyData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(weightsData, host_weightsData, weightsSpaceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(workSpaceData, host_workSpaceData, workSpaceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(reserveSpaceData, host_reserveSpaceData, reserveSpaceSize, cudaMemcpyHostToDevice);

    cudaMemcpy(dxData, host_dxData, sizeof(float) * x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dyData, host_dyData, sizeof(float) * y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhxData, host_dhxData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhyData, host_dhyData, sizeof(float) * h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcxData, host_dcxData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcyData, host_dcyData, sizeof(float) * c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dweightsData, host_dweightsData, weightsSpaceSize, cudaMemcpyHostToDevice);

    int *seqlenarray;
    cudaMalloc(&seqlenarray, sizeof(int) * 3);
    cudaMemcpy(seqlenarray, seqLenArray, sizeof(int) * 3, cudaMemcpyHostToDevice);

    auto e = cudnnRNNForward(
        handle, 
        rnnDesc, 
        CUDNN_FWD_MODE_TRAINING, 
        seqlenarray, 
        xDesc, 
        xData, 
        yDesc, 
        yData, 
        hDesc, 
        hxData, 
        hyData, 
        cDesc, 
        cxData,
        cyData,
        weightsSpaceSize, 
        weightsData, 
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    cudnnRNNBackwardData_v8(
        handle,
        rnnDesc,
        seqlenarray,
        yDesc,
        yData,
        dyData,
        xDesc,
        dxData,
        hDesc,
        hxData,
        dhyData,
        dhxData,
        cDesc,
        cxData,
        dcyData,
        dcxData,
        weightsSpaceSize,
        weightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    cudnnRNNBackwardWeights_v8(
        handle,
        rnnDesc,
        CUDNN_WGRAD_MODE_ADD,
        //CUDNN_WGRAD_MODE_SET,
        seqlenarray,
        xDesc,
        xData,
        hDesc,
        hxData,
        yDesc,
        yData,
        weightsSpaceSize,
        dweightsData,
        workSpaceSize, 
        workSpaceData, 
        reserveSpaceSize, 
        reserveSpaceData
    );

    cudaMemcpy(host_dxData, dxData, sizeof(float) * x_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dhxData, dhxData, sizeof(float) * h_size, cudaMemcpyDeviceToHost);
    std::vector<float> expect_dx = {
        1672, 1672,
        1672, 1672,
        1672, 1672,
        
        496, 496,
        496, 496,
        496, 496,
        
        124, 124,
        124, 124,
        124, 124,
        
        22, 22,
        22, 22,
        22, 22,
      };
    std::vector<float> expect_dhx = {
        1672, 1672,
        1672, 1672,
        1672, 1672,
        
        340, 340,
        340, 340,
        340, 340,
        
        46, 46,
        46, 46,
        46, 46,
      };
    check(expect_dx, host_dxData, expect_dx.size(), 1e-3);
    check(expect_dhx, host_dhxData, expect_dhx.size(), 1e-3);
    cudnnDestroy(handle);
}

int main() {
    test1<CUDNN_DATA_FLOAT>();
    test2<CUDNN_DATA_FLOAT>();
    test3<CUDNN_DATA_FLOAT>();
    std::cout << "test passed" << std::endl;
    return 0;
}