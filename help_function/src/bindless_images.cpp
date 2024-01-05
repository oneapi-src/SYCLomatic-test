// ===------------ bindless_images.cpp ---------- -*- C++ -* --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <dpct/dpct.hpp>

dpct::experimental::bindless_image_wrapper<sycl::float4, 4> w;

int main() {
  auto q = dpct::get_default_queue();
  sycl::ext::oneapi::experimental::image_descriptor d(
      {1, 1}, sycl::image_channel_order::rgba, sycl::image_channel_type::fp32);
  dpct::experimental::image_mem_ptr mem =
      new sycl::ext::oneapi::experimental::image_mem(d, q);
  sycl::float4 *data =
      (sycl::float4 *)dpct::dpct_malloc(sizeof(sycl::float4), q);
  dpct::image_data img(mem);
  img.set_data(mem);
  size_t pitch;
  sycl::float4 *data2d =
      (sycl::float4 *)dpct::dpct_malloc(pitch, sizeof(sycl::float4), 1);
  dpct::sampling_info samp;
  dpct::image_channel chan = dpct::image_channel::create<sycl::float4>();
  printf("prepare variable pass!\n");
  dpct::experimental::async_dpct_memcpy(mem, 0, 0, data, 1, 1, 1, q);
  dpct::experimental::dpct_memcpy(mem, 0, 0, data, 1, 1, 1, q);
  dpct::experimental::async_dpct_memcpy(mem, 0, 0, data, 1, q);
  dpct::experimental::dpct_memcpy(mem, 0, 0, data, 1, q);
  dpct::experimental::async_dpct_memcpy(data, mem, 0, 0, 1, 1, 1, q);
  dpct::experimental::dpct_memcpy(data, mem, 0, 0, 1, 1, 1, q);
  dpct::experimental::async_dpct_memcpy(data, mem, 0, 0, 1, q);
  dpct::experimental::dpct_memcpy(data, mem, 0, 0, 1, q);
  dpct::experimental::dpct_memcpy(mem, 0, 0, mem, 0, 0, 1, 1, q);
  dpct::experimental::dpct_memcpy(mem, 0, 0, mem, 0, 0, 1, q);
  printf("memory copy pass!\n");
  auto h = dpct::experimental::create_bindless_image(img, samp, q);
  dpct::experimental::get_data(h);
  dpct::experimental::get_sampling_info(h);
  dpct::experimental::get_channel(mem);
  dpct::experimental::destroy_bindless_image(h, q);
  printf("texture object pass!\n");
  w.attach(data, 1, chan, q);
  w.attach(data, 1, q);
  w.attach(data2d, 1, 1, pitch, chan, q);
  w.attach(data2d, 1, 1, pitch, chan, q);
  w.attach(mem, chan, q);
  w.attach(mem, q);
  w.detach(q);
  w.set_channel(chan);
  w.get_channel();
  w.set_channel_size(4, 1);
  w.get_channel_size();
  w.set_channel_data_type(dpct::image_channel_data_type::fp);
  w.get_channel_data_type();
  w.set(sycl::addressing_mode::repeat);
  w.get_addressing_mode();
  w.set(sycl::coordinate_normalization_mode::unnormalized);
  w.is_coordinate_normalized();
  w.set(sycl::filtering_mode::nearest);
  w.get_filtering_mode();
  w.get_handle();
  printf("texture referece pass!\n");
  printf("test pass!\n");
  return 0;
}
