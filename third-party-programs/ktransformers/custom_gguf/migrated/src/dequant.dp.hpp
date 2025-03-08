#ifndef DEQUANT_DP_H
#define DEQUANT_DP_H

#include <torch/torch.h>

torch::Tensor dequantize_q8_0(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);

torch::Tensor dequantize_q6_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);

torch::Tensor dequantize_q5_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);

torch::Tensor dequantize_q4_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);

torch::Tensor dequantize_q3_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);

torch::Tensor dequantize_q2_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);

                                                          
torch::Tensor dequantize_iq4_xs(const int8_t *data, const int num_bytes,
                                const int blk_size, const int ele_per_blk,
                                const torch::Device device,
                                const torch::Dtype target_dtype);



#endif // DEQUANT_DP_H