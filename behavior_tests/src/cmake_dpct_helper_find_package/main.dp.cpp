#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
#include <dpct/blas_utils.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

#include <cstdlib>

using data_type = double;
template <typename T>
bool validateCublasResult(T *expect, T *actual, int num, float precision) {
  for (int i = 0; i < num; i++) {
    if (std::abs(expect[i] - actual[i]) > precision) {
      std::cout << "Failed: at index - " << i;
      std::cout << ": " << actual[i] << " != " << expect[i];
      std::cout << std::endl;
      return false;
    }
  }
  return true;
}

bool cublasCheck() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  const int N = 3; // Matrix size (N x N)

    // Allocate host memory for matrices A, B, and C
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(N * N * sizeof(float));
    h_B = (float*)malloc(N * N * sizeof(float));
    h_C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    d_A = sycl::malloc_device<float>(N * N, q_ct1);
    d_B = sycl::malloc_device<float>(N * N, q_ct1);
    d_C = sycl::malloc_device<float>(N * N, q_ct1);

    // Copy matrices A and B from host to device
    q_ct1.memcpy(d_A, h_A, N * N * sizeof(float));
    q_ct1.memcpy(d_B, h_B, N * N * sizeof(float)).wait();

    // Create CUBLAS handle
    dpct::queue_ptr handle;
    handle = &q_ct1;

    const float alpha = 1.0f, beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    oneapi::mkl::blas::column_major::gemm(
        *handle, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, N, N, N,
        dpct::get_value(&alpha, *handle), d_A, N, d_B, N,
        dpct::get_value(&beta, *handle), d_C, N);

    // Copy matrix C from device to host
    q_ct1.memcpy(h_C, d_C, N * N * sizeof(float)).wait();

    // Allocate host memory for expected reference matrix C
    float h_ref_C[] = {0.671075, 0.718742, 0.866297, 1.06056, 1.35431, 1.3646, 1.00014, 1.33957, 1.36142};

    // Validate the result
    bool result = validateCublasResult(h_C, h_ref_C, N, 1e-5f);

    // Free allocated memory and destroy CUBLAS handle
    handle = nullptr;
    sycl::free(d_A, q_ct1);
    sycl::free(d_B, q_ct1);
    sycl::free(d_C, q_ct1);
    free(h_A);
    free(h_B);
    free(h_C);

    return result;
}


// Kernel to initialize the device array with some values
void initKernel(int *data, int size, const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < size) {
        data[idx] = idx;
    }
}

// Function to validate the result of the reduction
bool validateCubResult(int *hostData, int size, int result) {
    int hostResult = 0;
    for (int i = 0; i < size; ++i) {
        hostResult += hostData[i];
    }
    return hostResult == result;
}

bool cubCheck() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    const int size = 128; // Example array size
    int *deviceData, *deviceResult;
    int hostResult;

    // Allocate memory on the device
    deviceData = sycl::malloc_device<int>(size, q_ct1);
    deviceResult = sycl::malloc_device<int>(1, q_ct1);

    // Initialize the device array
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, (size + 31) / 32) *
                              sycl::range<3>(1, 1, 32),
                          sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            initKernel(deviceData, size, item_ct1);
        });
    dev_ct1.queues_wait_and_throw();

    // Allocate temporary storage

    /*
    DPCT1026:0: The call to cub::DeviceReduce::Sum was removed because this
    functionality is redundant in SYCL.
    */

    // Run the reduction
    q_ct1
        .fill(deviceResult,
              oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(q_ct1),
                                  deviceData, deviceData + size,
                                  typename std::iterator_traits<
                                      decltype(deviceResult)>::value_type{}),
              1)
        .wait();
    dev_ct1.queues_wait_and_throw();

    // Copy the result back to the host
    q_ct1.memcpy(&hostResult, deviceResult, sizeof(int));

    // Validate the result
    int *hostData = new int[size];
    q_ct1.memcpy(hostData, deviceData, size * sizeof(int)).wait();
    bool result = validateCubResult(hostData, size, hostResult);

    // Clean up
    delete[] hostData;
    sycl::free(deviceData, q_ct1);
    sycl::free(deviceResult, q_ct1);

    return result;
}

int main(int argc, char *argv[]) {
  bool cublas_check = cublasCheck();
  bool cub_check = cubCheck();

  std::cout << "CUBLAS check: " << (cublas_check? "Passed": "Failed")  << std::endl;
  std::cout << "CUB check: "    << (cub_check? "Passed": "Failed")     << std::endl;

  if (!(cublas_check && cub_check)) return 1;
  
  return 0;
}
