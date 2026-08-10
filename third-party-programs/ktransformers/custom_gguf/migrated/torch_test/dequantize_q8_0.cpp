#include <torch/torch.h>
#include <dequant.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <random>
#include <cstdint>

torch::Tensor dequantize_q8_0(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int num_blocks = num_bytes / blk_size;
    const c10::OptionalDeviceGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({ num_bytes }, options);

    q_ct1.memcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes).wait();

    // Create output tensor
    auto output = torch::zeros({ num_blocks, 32 }, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
      case torch::kFloat16: {
            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp16});

            q_ct1.submit([&](sycl::handler &cgh) {
                  const int8_t *data_gpu_data_ptr_int8_t_ct0 =
                      data_gpu.data_ptr<int8_t>();
                  auto output_data_ptr_ct1 = (sycl::half *)output.data_ptr();

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, 512) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_q8_0_fp16_kernel(
                                data_gpu_data_ptr_int8_t_ct0,
                                output_data_ptr_ct1, blk_size, ele_per_blk,
                                num_blocks, item_ct1);
                      });
            });
      } break;
      case torch::kBFloat16: {
            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp16});

            q_ct1.submit([&](sycl::handler &cgh) {
                  const int8_t *data_gpu_data_ptr_int8_t_ct0 =
                      data_gpu.data_ptr<int8_t>();
                  auto output_data_ptr_ct1 =
                      (sycl::ext::oneapi::bfloat16 *)output.data_ptr();

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, 512) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_q8_0_bf16_kernel(
                                data_gpu_data_ptr_int8_t_ct0,
                                output_data_ptr_ct1, blk_size, ele_per_blk,
                                num_blocks, item_ct1);
                      });
            });
      } break;
      case torch::kFloat32: {
            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp16});

            q_ct1.submit([&](sycl::handler &cgh) {
                  const int8_t *data_gpu_data_ptr_int8_t_ct0 =
                      data_gpu.data_ptr<int8_t>();
                  auto output_data_ptr_float_ct1 = output.data_ptr<float>();

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, 512) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_q8_0_fp32_kernel(
                                data_gpu_data_ptr_int8_t_ct0,
                                output_data_ptr_float_ct1, blk_size,
                                ele_per_blk, num_blocks, item_ct1);
                      });
            });
      } break;
        default:
            printf("target type not support\n");
            exit(0);
    }

    dev_ct1.queues_wait_and_throw();
    return output;
}


int main() {
    const int num_bytes = 1024;
    int8_t data[num_bytes];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);

    for (int i = 0; i < num_bytes; ++i) {
        data[i] = static_cast<int8_t>(dis(gen));
    }

    const int blk_size = 256;
    const int ele_per_blk = 32;
    const torch::Device device(torch::kXPU, 0);
    const torch::Dtype target_dtype = torch::kFloat32;

    torch::Tensor result = dequantize_q8_0(data, num_bytes, blk_size, ele_per_blk, device, target_dtype);

    std::cout << result << std::endl;

    return 0;
}
