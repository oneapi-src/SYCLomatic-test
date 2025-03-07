#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

void dequantize_q3_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {

    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);

        uint32_t aux[4];
        const int8_t * scales = (const int8_t*)aux;
        const float d_all =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 108)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ q  = (uint8_t*)(data + block_id * blk_size + 32);
        const uint8_t * __restrict__ hm = (uint8_t*)(data + block_id * blk_size + 0);
        uint8_t m = 1;


        uint8_t* block_scales = (uint8_t*)(data + block_id * blk_size + 96);

        for (int i = 0; i < 3; i++) {  
            aux[i] = 0;  
            for (int j = 0; j < 4; j++) {  
                aux[i] |= ((uint32_t)block_scales[i * 4 + j]) << (j * 8);
            }
        }

        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

int main() {
    // Define the parameters
    const int blk_size = 128 + 32 + 2 * sizeof(sycl::half); // Adjusted to match the kernel's data layout
    const int ele_per_blk = 256;
    const int num_blocks = 2;

    // Initialize input data
    std::vector<int8_t> data(blk_size * num_blocks);
    std::vector<float> output(ele_per_blk * num_blocks, 0.0f);

    // Fill the data with some values
    for (int i = 0; i < num_blocks; ++i) {
        sycl::half d_all = 0.5f;
        std::memcpy(data.data() + i * blk_size + 108, &d_all, sizeof(sycl::half));
        for (int j = 0; j < 32; ++j) {
            data[i * blk_size + j] = j % 2; // Initialize hm values
        }
        for (int j = 32; j < 128 + 32; ++j) {
            data[i * blk_size + j] = (j - 32) % 256; // Initialize q values
        }
        for (int j = 0; j < 16; ++j) {
            data[i * blk_size + 96 + j] = j % 16; // Initialize block scales
        }
    }

    // Create a SYCL queue
    queue q;

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
            dequantize_q3_k_fp32_kernel(d_data, d_output, blk_size, ele_per_blk, num_blocks, item_ct1);
        });
    }).wait();

    // Copy the result back to host
    q.memcpy(output.data(), d_output, output.size() * sizeof(float)).wait();

    // Free device memory
    free(d_data, q);
    free(d_output, q);

    // Check the results
    bool success = true;
    for (int i = 0; i < num_blocks; ++i) {
        sycl::half d_all = 0.5f;
        for (int j = 0; j < ele_per_blk; ++j) {
            // Calculate expected value
            int block_offset = i * blk_size;
            int q_offset = block_offset + 32 + (j / 128) * 32;
            int scale_offset = block_offset + 96 + (j / 64) * 2;
            uint8_t sc = data[scale_offset];
            float dl = d_all * (sc - 32);
            int q_idx = (j % 64) / 16;
            int shift = (j % 16) * 2;
            int8_t q_val = (data[q_offset + q_idx] >> shift) & 3;
            uint8_t hm_val = data[block_offset + (j % 32)];
            uint8_t m = 1 << (j % 8);
            float expected = dl * (q_val - ((hm_val & m) ? 0 : 4));

            if (std::fabs(output[i * ele_per_blk + j] - expected) > 1e-3) {
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
