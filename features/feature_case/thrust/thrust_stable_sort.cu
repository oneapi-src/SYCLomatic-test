// ====------ thrust_stable_sort.cu------------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-------------------------------------------------------------------------===//


#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>


void test_1() {
    const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={1, 2, 4, 5, 7, 8};
    thrust::host_vector<int> v(datas,datas+N);
   
    thrust::stable_sort(thrust::host, v.begin(), v.end());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_1 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_1 run passed!\n");
}

void test_2() {

   const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={1, 2, 4, 5, 7, 8};
    thrust::host_vector<int> v(datas,datas+N);
   
    thrust::stable_sort( v.begin(), v.end());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_2 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_2 run passed!\n");
}

void test_3() {
 const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={8, 7, 5, 4, 2, 1};
    thrust::host_vector<int> v(datas,datas+N);
   
    thrust::stable_sort(thrust::host, v.begin(), v.end(), thrust::greater<int>());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_3 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_3 run passed!\n");
}

void test_4() {

   const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={8, 7, 5, 4, 2, 1};
    thrust::host_vector<int> v(datas,datas+N);
   
    thrust::stable_sort( v.begin(), v.end(), thrust::greater<int>());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_4 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_4 run passed!\n");
}



void test_5() {
    const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={1, 2, 4, 5, 7, 8};
    thrust::device_vector<int> v(datas,datas+N);
   
    thrust::stable_sort(thrust::device, v.begin(), v.end());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_5 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_5 run passed!\n");
}

void test_6() {

   const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={1, 2, 4, 5, 7, 8};
    thrust::device_vector<int> v(datas,datas+N);
   
    thrust::stable_sort( v.begin(), v.end());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_6 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_6 run passed!\n");
}

void test_7() {
 const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={8, 7, 5, 4, 2, 1};
    thrust::device_vector<int> v(datas,datas+N);
   
    thrust::stable_sort(thrust::device, v.begin(), v.end(), thrust::greater<int>());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_7 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_7 run passed!\n");
}

void test_8() {

   const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={8, 7, 5, 4, 2, 1};
    thrust::device_vector<int> v(datas,datas+N);
   
    thrust::stable_sort( v.begin(), v.end(), thrust::greater<int>());
    for(int i=0;i<N;i++){
        if (v[i]!=ans[i]) {
            printf("test_8 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_8 run passed!\n");
}




void test_9() {
    const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={1, 2, 4, 5, 7, 8};
    
    thrust::stable_sort(thrust::host, datas,datas+N);
    for(int i=0;i<N;i++){
        if (datas[i]!=ans[i]) {
            printf("test_9 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_9 run passed!\n");
}

void test_10() {

   const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={1, 2, 4, 5, 7, 8};
       thrust::stable_sort( datas,datas+N);

    for(int i=0;i<N;i++){
        if (datas[i]!=ans[i]) {
            printf("test_10 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_10 run passed!\n");
}

void test_11() {
 const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={8, 7, 5, 4, 2, 1};
    
    thrust::stable_sort(thrust::host, datas,datas+N, thrust::greater<int>());
    for(int i=0;i<N;i++){
        if (datas[i]!=ans[i]) {
            printf("test_11 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_11 run passed!\n");
}

void test_12() {

 const int N=6;
    int datas[N]={1, 4, 2, 8, 5, 7};
    int ans[N]={8, 7, 5, 4, 2, 1};
    
    thrust::stable_sort( datas,datas+N, thrust::greater<int>());
    for(int i=0;i<N;i++){
        if (datas[i]!=ans[i]) {
            printf("test_12 run failed\n");
            exit(-1);
        }
    }
    

  printf("test_12 run passed!\n");
}




int main() {
  test_1();
  test_2();
  test_3();
  test_4();
  test_5();
  test_6();
  test_7();
  test_8();

  test_9();
  test_10();
  test_11();
  test_12();

  return 0;
}