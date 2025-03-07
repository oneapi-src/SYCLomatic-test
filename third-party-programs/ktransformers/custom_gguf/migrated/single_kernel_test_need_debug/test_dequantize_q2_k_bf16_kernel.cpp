#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;

void dequantize_q2_k_bf16_kernel(const int8_t *data,
                                 sycl::ext::oneapi::bfloat16 *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::ext::oneapi::bfloat16 *__restrict__ output_blk =
            (sycl::ext::oneapi::bfloat16 *)(output + block_id * ele_per_blk);

        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 80)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float min =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 82)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ q = (uint8_t*)(data + block_id * blk_size + 16);

        int is = 0;
        float dl, ml;

        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t* scales = (uint8_t*)(data + block_id * blk_size + (is++));
                uint8_t sc = *scales;
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ =
                    sycl::ext::oneapi::bfloat16(
                        dl * ((int8_t)((q[l] >> shift) & 3)) - ml);

                scales = (uint8_t*)(data + block_id * blk_size + (is++));
                sc = *scales;

                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ =
                    sycl::ext::oneapi::bfloat16(
                        dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml);

                shift += 2;
            }
            q += 32;
        }
    }
}

int main() {
    // Define the parameters
    const int blk_size = 128 + 16 + 2 * sizeof(sycl::half); // Adjusted to match the kernel's data layout
    const int ele_per_blk = 256;
    const int num_blocks = 2;

    // Initialize input data
    std::vector<int8_t> data(blk_size * num_blocks);
    std::vector<sycl::ext::oneapi::bfloat16> output(ele_per_blk * num_blocks, 0.0f);

    // Fill the data with some values
    for (int i = 0; i < num_blocks; ++i) {
        sycl::half d = 0.5f;
        sycl::half min = 0.1f;
        std::memcpy(data.data() + i * blk_size + 80, &d, sizeof(sycl::half));
        std::memcpy(data.data() + i * blk_size + 82, &min, sizeof(sycl::half));
        for (int j = 0; j < 16; ++j) {
            data[i * blk_size + j] = j;
        }
        for (int j = 16; j < 128 + 16; ++j) {
            data[i * blk_size + j] = (j - 16) % 256;
        }
    }

    // Create a SYCL queue
    queue q;

    // Allocate device memory
    int8_t* d_data = malloc_device<int8_t>(data.size(), q);
    sycl::ext::oneapi::bfloat16* d_output = malloc_device<sycl::ext::oneapi::bfloat16>(output.size(), q);

    // Copy data to device
    q.memcpy(d_data, data.data(), data.size() * sizeof(int8_t)).wait();
    q.memcpy(d_output, output.data(), output.size() * sizeof(sycl::ext::oneapi::bfloat16)).wait();

    // Define the kernel execution configuration
    range<3> global_work_size(1, 1, num_blocks);
    range<3> local_work_size(1, 1, 1);

    // Launch the kernel
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<3>(global_work_size, local_work_size), [=](nd_item<3> item_ct1) {
            dequantize_q2_k_bf16_kernel(d_data, d_output, blk_size, ele_per_blk, num_blocks, item_ct1);
        });
    }).wait();

    // Copy the result back to host
    q.memcpy(output.data(), d_output, output.size() * sizeof(sycl::ext::oneapi::bfloat16)).wait();

    // Free device memory
    free(d_data, q);
    free(d_output, q);

    // Check the results
    bool success = true;
    for (int i = 0; i < num_blocks; ++i) {
        sycl::half d = 0.5f;
        sycl::half min = 0.1f;
        for (int j = 0; j < ele_per_blk; ++j) {
            // Calculate expected value
            int block_offset = i * blk_size;
            int q_offset = block_offset + 16 + (j / 128) * 32;
            int scale_offset = block_offset + (j / 64) * 2;
            uint8_t sc = data[scale_offset];
            float dl = d * (sc & 0xF);
            float ml = min * (sc >> 4);
            int q_idx = (j % 64) / 16;
            int shift = (j % 16) * 2;
            int8_t q_val = (data[q_offset + q_idx] >> shift) & 3;
            float expected = dl * q_val - ml;
            sycl::ext::oneapi::bfloat16 expected_bf16 = sycl::ext::oneapi::bfloat16(expected);

            if (std::fabs(static_cast<float>(output[i * ele_per_blk + j]) - static_cast<float>(expected_bf16)) > 1e-3) {
                success = false;
                std::cout << "Mismatch at block " << i << ", element " << j << ": expected " << static_cast<float>(expected_bf16) << ", got " << static_cast<float>(output[i * ele_per_blk + j]) << std::endl;
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
