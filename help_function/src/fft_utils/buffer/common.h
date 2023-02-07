// ===--- common.h --------------------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#pragma once

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>

template<class T>
void print_values(T* ptr, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    std::cout << ptr[i] << ", " << std::endl;
  }
}
template <> void print_values(sycl::float2 *ptr, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    std::cout << "(" << ptr[i].x() << "," << ptr[i].y() << "), " << std::endl;
  }
}
template <> void print_values(sycl::double2 *ptr, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    std::cout << "(" << ptr[i].x() << "," << ptr[i].y() << "), " << std::endl;
  }
}

template<class T>
void print_values(T* ptr, std::vector<int> indices) {
  for (int i = 0 ; i < indices.size(); i++) {
    std::cout << ptr[indices[i]] << ", " << std::endl;
  }
}
template <> void print_values(sycl::float2 *ptr, std::vector<int> indices) {
  for (int i = 0 ; i < indices.size(); i++) {
    std::cout << "(" << ptr[indices[i]].x() << "," << ptr[indices[i]].y()
              << "), " << std::endl;
  }
}
template <> void print_values(sycl::double2 *ptr, std::vector<int> indices) {
  for (int i = 0 ; i < indices.size(); i++) {
    std::cout << "(" << ptr[indices[i]].x() << "," << ptr[indices[i]].y()
              << "), " << std::endl;
  }
}

template<class T>
bool compare(T* ptr1, T* ptr2, std::vector<int> indices) {
  for (int i = 0 ; i < indices.size(); i++) {
    if (std::abs(ptr1[indices[i]] - ptr2[indices[i]]) > 0.01) {
      return false;
    }
  }
  return true;
}
template <>
bool compare(sycl::float2 *ptr1, sycl::float2 *ptr2, std::vector<int> indices) {
  for (int i = 0 ; i < indices.size(); i++) {
    if (std::abs(ptr1[indices[i]].x() - ptr2[indices[i]].x()) > 0.01 ||
        std::abs(ptr1[indices[i]].y() - ptr2[indices[i]].y()) > 0.01) {
      return false;
    }
  }
  return true;
}
template <>
bool compare(sycl::double2 *ptr1, sycl::double2 *ptr2,
             std::vector<int> indices) {
  for (int i = 0 ; i < indices.size(); i++) {
    if (std::abs(ptr1[indices[i]].x() - ptr2[indices[i]].x()) > 0.01 ||
        std::abs(ptr1[indices[i]].y() - ptr2[indices[i]].y()) > 0.01) {
      return false;
    }
  }
  return true;
}

template<class T>
bool compare(T* ptr1, T* ptr2, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    if (std::abs(ptr1[i] - ptr2[i]) > 0.01) {
      return false;
    }
  }
  return true;
}
template <> bool compare(sycl::float2 *ptr1, sycl::float2 *ptr2, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    if (std::abs(ptr1[i].x() - ptr2[i].x()) > 0.01 ||
        std::abs(ptr1[i].y() - ptr2[i].y()) > 0.01) {
      return false;
    }
  }
  return true;
}
template <>
bool compare(sycl::double2 *ptr1, sycl::double2 *ptr2, int ele_num) {
  for (int i = 0 ; i < ele_num; i++) {
    if (std::abs(ptr1[i].x() - ptr2[i].x()) > 0.01 ||
        std::abs(ptr1[i].y() - ptr2[i].y()) > 0.01) {
      return false;
    }
  }
  return true;
}

template <class T>
void set_value(T* data_ptr, int ele_num) {
  for (int i = 0; i < ele_num; i++) {
    data_ptr[i] = i;
  }
}

template <class T>
void set_value_with_stride(T* data_ptr, int ele_num, int stride) {
  int value = 0;
  for (int i = 0; i < ele_num * stride; i++) {
    if (i % stride == 0) {
      data_ptr[i] = value;
      value++;
    }
  }
}

template <>
void set_value_with_stride(sycl::float2 *data_ptr, int ele_num, int stride) {
  int value = 0;
  float* data_ptr_float = (float*)data_ptr;
  for (int i = 0; i < ele_num * stride * 2; i++) {
    if ((i % (stride * 2) == 0) || (i % (stride * 2) == 1)) {
      data_ptr_float[i] = value;
      value++;
    }
  }
}

template <>
void set_value_with_stride(sycl::double2 *data_ptr, int ele_num, int stride) {
  int value = 0;
  double* data_ptr_double = (double*)data_ptr;
  for (int i = 0; i < ele_num * stride * 2; i++) {
    if ((i % (stride * 2) == 0) || (i % (stride * 2) == 1)) {
      data_ptr_double[i] = value;
      value++;
    }
  }
}

template <class T>
void set_value(T* data_ptr, int height, int width, int pitch) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < pitch; j++) {
      if (j < width)
        data_ptr[i * pitch + j] = i * width + j;
      else
        data_ptr[i * pitch + j] = 0;
    }
  }
}

template <class T>
void set_value(T* data_ptr, int depth, int height, int width, int pitch) {
  for (int d = 0; d < depth; d++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < pitch; j++) {
        int idx = (height * pitch) * d + pitch * i + j;
        if (j < width)
          data_ptr[idx] = (height * width) * d + width * i + j;
        else
          data_ptr[idx] = 0;
      }
    }
  }
}

#define TO_STRING(ARG) #ARG
