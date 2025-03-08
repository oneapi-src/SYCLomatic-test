#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
 
using namespace sycl;
 
void dequantize_q8_0_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        const int8_t* cur_block = data + block_id * blk_size;
        float scale = sycl::vec<sycl::half, 1>(*((sycl::half *)cur_block))
                          .convert<float, sycl::rounding_mode::automatic>()[0];
        cur_block += 2;
        for (int i = 0; i < ele_per_blk; i++){
            output_blk[i] = scale * cur_block[i];
        }
    }
}
 
int main() {
    // Define the parameters
    const int blk_size = 10;
    const int ele_per_blk = 8;
    const int num_blocks = 2;
 
    // Initialize input data
    std::vector<int8_t> data(blk_size * num_blocks);
    std::vector<float> output(ele_per_blk * num_blocks, 0.0f);
 
    // Fill the data with some values
    for (int i = 0; i < num_blocks; ++i) {
        sycl::half scale = 0.5f;
        std::memcpy(data.data() + i * blk_size, &scale, sizeof(sycl::half));
        for (int j = 2; j < blk_size; ++j) {
            data[i * blk_size + j] = j - 2;
        }
    }

        // Print the input data
    std::cout << "Input data:" << std::endl;
    for (int i = 0; i < num_blocks; ++i) {
        std::cout << "Block " << i << ":" << std::endl;
        for (int j = 0; j < blk_size; ++j) {
            std::cout << static_cast<int>(data[i * blk_size + j]) << " ";
        }
        std::cout << std::endl;
    }

 
    // Create a SYCL queue
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
 
    // Allocate device memory
    int8_t* d_data = malloc_device<int8_t>(data.size(), q);
    float* d_output = malloc_device<float>(output.size(), q);
 
    // Copy data to device
    q.memcpy(d_data, data.data(), data.size() * sizeof(int8_t)).wait();
    q.memcpy(d_output, output.data(), output.size() * sizeof(float)).wait();
 
    // Define the kernel execution configuration
    range<3> global_work_size(1, 1, num_blocks);
    range<3> local_work_size(1, 1, 1);
 
    // Launch the kernel
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<3>(global_work_size, local_work_size), [=](nd_item<3> item_ct1) {
            dequantize_q8_0_fp32_kernel(d_data, d_output, blk_size, ele_per_blk, num_blocks, item_ct1);
        });
    }).wait();
 
    // Copy the result back to host
    q.memcpy(output.data(), d_output, output.size() * sizeof(float)).wait();
 
    // Free device memory
    free(d_data, q);
    free(d_output, q);

        // Print the output data
    std::cout << "Output data:" << std::endl;
    for (int i = 0; i < num_blocks; ++i) {
        std::cout << "Block " << i << ":" << std::endl;
        for (int j = 0; j < ele_per_blk; ++j) {
            std::cout << output[i * ele_per_blk + j] << " ";
        }
        std::cout << std::endl;
    }

 
    // Check the results
    bool success = true;
    for (int i = 0; i < num_blocks; ++i) {
        sycl::half scale = 0.5f;
        for (int j = 0; j < ele_per_blk; ++j) {
            float expected = scale * (j);
            if (std::fabs(output[i * ele_per_blk + j] - expected) > 1e-5) {
                success = false;
                std::cout << "Mismatch at block " << i << ", element " << j << ": expected " << expected << ", got " << output[i * ele_per_blk + j] << std::endl;
            }
        }
    }
 
    if (success) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
 
    return 0;
}