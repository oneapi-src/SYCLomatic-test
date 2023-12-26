// ====------ thrust-gather.cu---------- *- CUDA -* ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>

struct is_odd
{
  __host__ __device__
  bool operator()(const int x) const
  {
    return (x % 2) == 1;
  }
};

void test_1() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  thrust::device_vector<int> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);  
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10,123);

  thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::device_vector<int> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);    
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<int> d_output(10,123);
  thrust::gather_if(thrust::device, d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  printf("test_2 run passed!\n");
}

void test_3() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<int> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);    
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10,123);
  thrust::gather_if(thrust::seq, h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }
  printf("test_3 run passed!\n");
}

void test_4() {
  int values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<int> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<int> h_output(10,123);
  thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  int values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  printf("test_4 run passed!\n");
}


template<typename T>
void test_A() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  thrust::device_vector<T> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);    
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<T> d_output(10,123);

  thrust::gather_if(d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};
  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_A run failed\n");
      exit(-1);
    }
  }

  printf("test_A run passed %ld!\n",sizeof(T));
}

template<typename T>
void test_B() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::device_vector<T> d_values(values, values + 10);
  thrust::device_vector<int> d_stencil(stencil, stencil + 10);      
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::device_vector<int> d_map(map, map + 10);
  thrust::device_vector<T> d_output(10,123);
  thrust::gather_if(thrust::device, d_map.begin(), d_map.end(), d_stencil.begin(), d_values.begin(),
                 d_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < d_output.size(); i++) {
    if (d_output[i] != values_ref[i]) {
      printf("test_B run failed\n");
      exit(-1);
    }
  }
  printf("test_B run passed %ld!\n",sizeof(T));
}

template<typename T>
void test_C() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<T> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);      
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<T> h_output(10,123);
  thrust::gather_if(thrust::seq, h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_C run failed\n");
      exit(-1);
    }
  }
  printf("test_C run passed %ld!\n",sizeof(T));
}

template<typename T>
void test_D() {
  T values[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  int stencil[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  
  thrust::host_vector<T> h_values(values, values + 10);
  thrust::host_vector<int> h_stencil(stencil, stencil + 10);
  int map[10] = {9, 8, 7, 6, 5, 0, 1, 2, 3, 4};
  thrust::host_vector<int> h_map(map, map + 10);
  thrust::host_vector<T> h_output(10,123);
  thrust::gather_if(h_map.begin(), h_map.end(), h_stencil.begin(), h_values.begin(),
                 h_output.begin(),is_odd());

  T values_ref[10] = {123, 9, 123, 7, 123, 1, 123, 3, 123, 5};

  for (int i = 0; i < h_output.size(); i++) {
    if (h_output[i] != values_ref[i]) {
      printf("test_D run failed\n");
      exit(-1);
    }
  }
  printf("test_D run passed %ld!\n",sizeof(T));
}

int main() {
  test_1();
  test_2();
  test_3();
  test_4();

  std::cout << "Testing char\n";
  test_A<char>();
  test_B<char>();
  test_C<char>();
  test_D<char>();

  std::cout << "Testing short int\n";
  test_A<short int>();
  test_B<short int>();
  test_C<short int>();
  test_D<short int>();

  std::cout << "Testing int\n";
  test_A<int>();
  test_B<int>();
  test_C<int>();
  test_D<int>();

  std::cout << "Testing long\n";
  test_A<long>();
  test_B<long>();
  test_C<long>();
  test_D<long>();

  std::cout << "Testing long long\n";
  test_A<long long>();
  test_B<long long>();
  test_C<long long>();
  test_D<long long>();
  
  std::cout << "Testing float\n";
  test_A<float>();
  test_B<float>();
  test_C<float>();
  test_D<float>();
  
  return 0;
}
