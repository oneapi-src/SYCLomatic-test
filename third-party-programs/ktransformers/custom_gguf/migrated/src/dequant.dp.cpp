/*
 * @Description  :  
 * @Author       : Azure-Tang, Boxin Zhang
 * @Date         : 2024-07-25 13:38:30
 * @Version      : 0.2.2
 * Adapted from https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c
 * Copyright (c) 2023-2024 The ggml authors
 * Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
 */

#include "dequant.dp.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdint>
#include <c10/core/DeviceGuard.h>

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

void dequantize_q8_0_fp16_kernel(const int8_t *data, sycl::half *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::half *__restrict__ output_blk =
            (sycl::half *)(output + block_id * ele_per_blk);
        const int8_t* cur_block = data + block_id * blk_size;
        float scale = sycl::vec<sycl::half, 1>(*((sycl::half *)cur_block))
                          .convert<float, sycl::rounding_mode::automatic>()[0];
        cur_block += 2;
        for (int i = 0; i < ele_per_blk; i++) {
            output_blk[i] =
                sycl::vec<float, 1>(scale * cur_block[i])
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        }
    }
}

void dequantize_q8_0_bf16_kernel(const int8_t *data,
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
        const int8_t* cur_block = data + block_id * blk_size;
        float scale = sycl::vec<sycl::half, 1>(*((sycl::half *)cur_block))
                          .convert<float, sycl::rounding_mode::automatic>()[0];
        cur_block += 2;
        for (int i = 0; i < ele_per_blk; i++) {
            output_blk[i] = sycl::ext::oneapi::bfloat16(scale * cur_block[i]);
        }
    }
}

// __device__ void get_scale_min_k4(int j, const uint8_t * __restrict__ q, uint8_t * __restrict__ d, uint8_t * __restrict__ m) {
void get_scale_min_k4(int j, const uint8_t * q, uint8_t * __restrict__ d, uint8_t * __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_q2_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);

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
                for (int l = 0; l < 16; ++l) *output_blk++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                scales = (uint8_t*)(data + block_id * blk_size + (is++));
                sc = *scales;

                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_q2_k_fp16_kernel(const int8_t *data, sycl::half *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::half *__restrict__ output_blk =
            (sycl::half *)(output + block_id * ele_per_blk);

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
                    sycl::vec<float, 1>(dl * ((int8_t)((q[l] >> shift) & 3)) -
                                        ml)
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];

                scales = (uint8_t*)(data + block_id * blk_size + (is++));
                sc = *scales;

                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *output_blk++ =
                    sycl::vec<float, 1>(
                        dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml)
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];

                shift += 2;
            }
            q += 32;
        }
    }
}

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

void dequantize_q3_k_fp16_kernel(const int8_t *data, sycl::half *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {

    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::half *__restrict__ output_blk =
            (sycl::half *)(output + block_id * ele_per_blk);

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
                    *output_blk++ =
                        sycl::vec<float, 1>(dl *
                                            ((int8_t)((q[l + 0] >> shift) & 3) -
                                             ((hm[l + 0] & m) ? 0 : 4)))
                            .convert<sycl::half,
                                     sycl::rounding_mode::automatic>()[0];
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ =
                        sycl::vec<float, 1>(
                            dl * ((int8_t)((q[l + 16] >> shift) & 3) -
                                  ((hm[l + 16] & m) ? 0 : 4)))
                            .convert<sycl::half,
                                     sycl::rounding_mode::automatic>()[0];
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_q3_k_bf16_kernel(const int8_t *data,
                                 sycl::ext::oneapi::bfloat16 *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {

    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::ext::oneapi::bfloat16 *__restrict__ output_blk =
            (sycl::ext::oneapi::bfloat16 *)(output + block_id * ele_per_blk);

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
                    *output_blk++ = sycl::ext::oneapi::bfloat16(
                        dl * ((int8_t)((q[l + 0] >> shift) & 3) -
                              ((hm[l + 0] & m) ? 0 : 4)));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *output_blk++ = sycl::ext::oneapi::bfloat16(
                        dl * ((int8_t)((q[l + 16] >> shift) & 3) -
                              ((hm[l + 16] & m) ? 0 : 4)));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}


void dequantize_q4_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        // const uint8_t * q = data[i].qs;
        const uint8_t * q = (uint8_t*)(data + block_id * 144 + 16);

        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * 144 + 0)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float min =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * 144 + 2)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < ele_per_blk; j += 64) {
            uint8_t* scales = (uint8_t*)(data + block_id * 144 + 4);
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *output_blk++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_q4_k_fp16_kernel(const int8_t *data, sycl::half *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::half *__restrict__ output_blk =
            (sycl::half *)(output + block_id * ele_per_blk);
        // const uint8_t * q = data[i].qs;
        const uint8_t * q = (uint8_t*)(data + block_id * 144 + 16);

        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * 144 + 0)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float min =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * 144 + 2)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < ele_per_blk; j += 64) {
            uint8_t* scales = (uint8_t*)(data + block_id * 144 + 4);
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::vec<float, 1>(d1 * (q[l] & 0xF) - m1)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::vec<float, 1>(d2 * (q[l] >> 4) - m2)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
            q += 32; is += 2;
        }
    }
}

void dequantize_q4_k_bf16_kernel(const int8_t *data,
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
        // const uint8_t * q = data[i].qs;
        const uint8_t * q = (uint8_t*)(data + block_id * 144 + 16);

        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * 144 + 0)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float min =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * 144 + 2)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < ele_per_blk; j += 64) {
            uint8_t* scales = (uint8_t*)(data + block_id * 144 + 4);
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::ext::oneapi::bfloat16(d1 * (q[l] & 0xF) - m1);
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::ext::oneapi::bfloat16(d2 * (q[l] >> 4) - m2);
            q += 32; is += 2;
        }
    }
}

void dequantize_q5_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);

        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 0)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float min =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 2)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 16);
        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size + 48);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 4);

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *output_blk++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_q5_k_fp16_kernel(const int8_t *data, sycl::half *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::half *__restrict__ output_blk =
            (sycl::half *)(output + block_id * ele_per_blk);

        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 0)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float min =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 2)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 16);
        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size + 48);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 4);

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::vec<float, 1>(
                    d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::vec<float, 1>(
                    d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_q5_k_bf16_kernel(const int8_t *data,
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
                                         data + block_id * blk_size + 0)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float min =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 2)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 16);
        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size + 48);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        uint8_t* scales = (uint8_t*)(data + block_id * blk_size + 4);

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::ext::oneapi::bfloat16(
                    d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1);
            for (int l = 0; l < 32; ++l) *output_blk++ =
                sycl::ext::oneapi::bfloat16(
                    d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2);
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_q6_k_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 208)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size);
        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 128);
        const int8_t  * __restrict__ sc = (int8_t*)(data + block_id * blk_size + 192);


        for (int n = 0; n < ele_per_blk; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                output_blk[l +  0] = d * sc[is + 0] * q1;
                output_blk[l + 32] = d * sc[is + 2] * q2;
                output_blk[l + 64] = d * sc[is + 4] * q3;
                output_blk[l + 96] = d * sc[is + 6] * q4;
            }
            output_blk += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_q6_k_fp16_kernel(const int8_t *data, sycl::half *output,
                                 const int blk_size, const int ele_per_blk,
                                 const int num_blocks,
                                 const sycl::nd_item<3> &item_ct1) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::half *__restrict__ output_blk =
            (sycl::half *)(output + block_id * ele_per_blk);
        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size + 208)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size);
        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 128);
        const int8_t  * __restrict__ sc = (int8_t*)(data + block_id * blk_size + 192);


        for (int n = 0; n < ele_per_blk; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                output_blk[l + 0] =
                    sycl::vec<float, 1>(d * sc[is + 0] * q1)
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];
                output_blk[l + 32] =
                    sycl::vec<float, 1>(d * sc[is + 2] * q2)
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];
                output_blk[l + 64] =
                    sycl::vec<float, 1>(d * sc[is + 4] * q3)
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];
                output_blk[l + 96] =
                    sycl::vec<float, 1>(d * sc[is + 6] * q4)
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];
            }
            output_blk += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_q6_k_bf16_kernel(const int8_t *data,
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
                                         data + block_id * blk_size + 208)))
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const uint8_t * __restrict__ ql = (uint8_t*)(data + block_id * blk_size);
        const uint8_t * __restrict__ qh = (uint8_t*)(data + block_id * blk_size + 128);
        const int8_t  * __restrict__ sc = (int8_t*)(data + block_id * blk_size + 192);


        for (int n = 0; n < ele_per_blk; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                output_blk[l + 0] =
                    sycl::ext::oneapi::bfloat16(d * sc[is + 0] * q1);
                output_blk[l + 32] =
                    sycl::ext::oneapi::bfloat16(d * sc[is + 2] * q2);
                output_blk[l + 64] =
                    sycl::ext::oneapi::bfloat16(d * sc[is + 4] * q3);
                output_blk[l + 96] =
                    sycl::ext::oneapi::bfloat16(d * sc[is + 6] * q4);
            }
            output_blk += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static dpct::global_memory<const int8_t, 1>
    kvalues_iq4nl(sycl::range<1>(16), {-127, -104, -83, -65, -49, -35, -22, -10,
                                       1, 13, 25, 38, 53, 69, 89, 113});

void dequantize_iq4_xs_fp32_kernel(const int8_t* data, float* output, const int blk_size, const int ele_per_blk, const int num_blocks,
                                   const sycl::nd_item<3> &item_ct1,
                                   const int8_t *kvalues_iq4nl) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        float* __restrict__ output_blk = (float*)(output + block_id * ele_per_blk);
        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const uint16_t scales_h = *(reinterpret_cast<const uint16_t*>(data + block_id * blk_size + 2));
        const uint8_t* scales_l = (uint8_t*)(data + block_id * blk_size + 2 + 2);
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2 + 2 + 4);

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                output_blk[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                output_blk[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            output_blk += 32;
            qs += 16;
        }
    }
}

void dequantize_iq4_xs_fp16_kernel(const int8_t *data, sycl::half *output,
                                   const int blk_size, const int ele_per_blk,
                                   const int num_blocks,
                                   const sycl::nd_item<3> &item_ct1,
                                   const int8_t *kvalues_iq4nl) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::half *__restrict__ output_blk =
            (sycl::half *)(output + block_id * ele_per_blk);
        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const uint16_t scales_h = *(reinterpret_cast<const uint16_t*>(data + block_id * blk_size + 2));
        const uint8_t* scales_l = (uint8_t*)(data + block_id * blk_size + 2 + 2);
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2 + 2 + 4);

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                output_blk[j + 0] =
                    sycl::vec<float, 1>(dl * kvalues_iq4nl[qs[j] & 0xf])
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];
                output_blk[j + 16] =
                    sycl::vec<float, 1>(dl * kvalues_iq4nl[qs[j] >> 4])
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0];
            }
            output_blk += 32;
            qs += 16;
        }
    }
}

void dequantize_iq4_xs_bf16_kernel(const int8_t *data,
                                   sycl::ext::oneapi::bfloat16 *output,
                                   const int blk_size, const int ele_per_blk,
                                   const int num_blocks,
                                   const sycl::nd_item<3> &item_ct1,
                                   const int8_t *kvalues_iq4nl) {
    long long global_idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                           item_ct1.get_local_id(2);
    for (long long block_id = global_idx; block_id < num_blocks;
         block_id +=
         item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        sycl::ext::oneapi::bfloat16 *__restrict__ output_blk =
            (sycl::ext::oneapi::bfloat16 *)(output + block_id * ele_per_blk);
        const float d =
            sycl::vec<sycl::half, 1>(*(reinterpret_cast<const sycl::half *>(
                                         data + block_id * blk_size)))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const uint16_t scales_h = *(reinterpret_cast<const uint16_t*>(data + block_id * blk_size + 2));
        const uint8_t* scales_l = (uint8_t*)(data + block_id * blk_size + 2 + 2);
        const uint8_t* qs = (uint8_t*)(data + block_id * blk_size + 2 + 2 + 4);

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                output_blk[j + 0] = sycl::ext::oneapi::bfloat16(
                    dl * kvalues_iq4nl[qs[j] & 0xf]);
                output_blk[j + 16] =
                    sycl::ext::oneapi::bfloat16(dl * kvalues_iq4nl[qs[j] >> 4]);
            }
            output_blk += 32;
            qs += 16;
        }
    }
}

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
    //data_gpu.copy_(data, false);

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

torch::Tensor dequantize_q6_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    // data.numel%blk_size should be 0, else raise err
    int num_blocks = num_bytes / blk_size;

    const c10::OptionalDeviceGuard device_guard(device);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    q_ct1.memcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes).wait();
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

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
                            dequantize_q6_k_fp16_kernel(
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
                            dequantize_q6_k_bf16_kernel(
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
                            dequantize_q6_k_fp32_kernel(
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

torch::Tensor dequantize_q5_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int num_blocks = num_bytes / blk_size;
    const c10::OptionalDeviceGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    q_ct1.memcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes).wait();
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

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
                            dequantize_q5_k_fp16_kernel(
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
                            dequantize_q5_k_bf16_kernel(
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
                            dequantize_q5_k_fp32_kernel(
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

torch::Tensor dequantize_q4_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    // data.numel%blk_size should be 0, else raise err
    int num_blocks = num_bytes / blk_size;
    const c10::OptionalDeviceGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    q_ct1.memcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes).wait();
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

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
                            dequantize_q4_k_fp16_kernel(
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
                            dequantize_q4_k_bf16_kernel(
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
                            dequantize_q4_k_fp32_kernel(
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

torch::Tensor dequantize_q3_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int num_blocks = num_bytes / blk_size;
    const c10::OptionalDeviceGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    q_ct1.memcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes).wait();
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

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
                            dequantize_q3_k_fp16_kernel(
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
                            dequantize_q3_k_bf16_kernel(
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
                            dequantize_q3_k_fp32_kernel(
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

torch::Tensor dequantize_q2_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int num_blocks = num_bytes / blk_size;
    const c10::OptionalDeviceGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    q_ct1.memcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes).wait();
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

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
                            dequantize_q2_k_fp16_kernel(
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
                            dequantize_q2_k_bf16_kernel(
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
                            dequantize_q2_k_fp32_kernel(
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

torch::Tensor dequantize_iq4_xs(const int8_t *data, const int num_bytes,
                                const int blk_size, const int ele_per_blk,
                                const torch::Device device,
                                const torch::Dtype target_dtype) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int num_blocks = num_bytes / blk_size;
    const c10::OptionalDeviceGuard device_guard(device);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto data_gpu = torch::empty({num_bytes}, options);

    q_ct1.memcpy(data_gpu.data_ptr<int8_t>(), data, num_bytes).wait();
    //data_gpu.copy_(data, false);

    // Create output tensor
    auto output = torch::zeros({num_blocks, 256}, torch::dtype(target_dtype).device(device));

    switch (target_dtype) {
      case torch::kFloat16: {
            kvalues_iq4nl.init();

            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp16});

            q_ct1.submit([&](sycl::handler &cgh) {
                  auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                  const int8_t *data_gpu_data_ptr_int8_t_ct0 =
                      data_gpu.data_ptr<int8_t>();
                  auto output_data_ptr_ct1 = (sycl::half *)output.data_ptr();

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, 512) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_iq4_xs_fp16_kernel(
                                data_gpu_data_ptr_int8_t_ct0,
                                output_data_ptr_ct1, blk_size, ele_per_blk,
                                num_blocks, item_ct1, kvalues_iq4nl_ptr_ct1);
                      });
            });
      } break;
      case torch::kBFloat16: {
            kvalues_iq4nl.init();

            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp16});

            q_ct1.submit([&](sycl::handler &cgh) {
                  auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                  const int8_t *data_gpu_data_ptr_int8_t_ct0 =
                      data_gpu.data_ptr<int8_t>();
                  auto output_data_ptr_ct1 =
                      (sycl::ext::oneapi::bfloat16 *)output.data_ptr();

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, 512) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_iq4_xs_bf16_kernel(
                                data_gpu_data_ptr_int8_t_ct0,
                                output_data_ptr_ct1, blk_size, ele_per_blk,
                                num_blocks, item_ct1, kvalues_iq4nl_ptr_ct1);
                      });
            });
      } break;
      case torch::kFloat32: {
            kvalues_iq4nl.init();

            dpct::has_capability_or_fail(q_ct1.get_device(),
                                         {sycl::aspect::fp16});

            q_ct1.submit([&](sycl::handler &cgh) {
                  auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                  const int8_t *data_gpu_data_ptr_int8_t_ct0 =
                      data_gpu.data_ptr<int8_t>();
                  auto output_data_ptr_float_ct1 = output.data_ptr<float>();

                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, 512) *
                                            sycl::range<3>(1, 1, 256),
                                        sycl::range<3>(1, 1, 256)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_iq4_xs_fp32_kernel(
                                data_gpu_data_ptr_int8_t_ct0,
                                output_data_ptr_float_ct1, blk_size,
                                ele_per_blk, num_blocks, item_ct1,
                                kvalues_iq4nl_ptr_ct1);
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
