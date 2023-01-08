// ====------ thrust_transform_exclusive_scan.cu------------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===--------------------------------------------------------------------------------------===//


#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

void test_1() {  //host iterator
    const int N=6;
    int A[N]={1, 0, 2, 2, 1, 3};
    int ans[N]={4, 3, 3, 1, -1, -2};
    thrust::host_vector<int> V(A,A+N);
   
    thrust::negate<int> unary_op;
    thrust::plus<int> binary_op;
    thrust::transform_exclusive_scan(thrust::host, V.begin(), V.end(), V.begin(), unary_op, 4, binary_op); // in-place scan
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_1 run failed\n");
            exit(-1);
        }
    }

  printf("test_1 run passed!\n");
}

void test_2() { //host iterator

   const int N=6;
    int A[N]={1, 0, 2, 2, 1, 3};
    int ans[N]={4, 3, 3, 1, -1, -2};
    thrust::host_vector<int> V(A,A+N);
   
    thrust::negate<int> unary_op;
    thrust::plus<int> binary_op;
    thrust::transform_exclusive_scan( V.begin(), V.end(), V.begin(), unary_op, 4, binary_op); // in-place scan
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_2 run failed\n");
            exit(-1);
        }
    }

  printf("test_2 run passed!\n");
}

void test_3() { //device iterator
   const int N=6;
    int A[N]={1, 0, 2, 2, 1, 3};
    int ans[N]={4, 3, 3, 1, -1, -2};
    thrust::device_vector<int> V(A,A+N);
   
    thrust::negate<int> unary_op;
    thrust::plus<int> binary_op;
    thrust::transform_exclusive_scan(thrust::device, V.begin(), V.end(), V.begin(), unary_op, 4, binary_op); // in-place scan
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_3 run failed\n");
            exit(-1);
        }
    }

  printf("test_3 run passed!\n");
}

void test_4() { //device iterator

   const int N=6;
    int A[N]={1, 0, 2, 2, 1, 3};
    int ans[N]={4, 3, 3, 1, -1, -2};
    thrust::device_vector<int> V(A,A+N);
   
    thrust::negate<int> unary_op;
    thrust::plus<int> binary_op;
    thrust::transform_exclusive_scan( V.begin(), V.end(), V.begin(), unary_op, 4, binary_op); // in-place scan
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_4 run failed\n");
            exit(-1);
        }
    }

  printf("test_4 run passed!\n");
}



void test_5() { //raw ptr
    const int N=6;
    int A[N]={1, 0, 2, 2, 1, 3};
    int ans[N]={4, 3, 3, 1, -1, -2};
   
    thrust::negate<int> unary_op;
    thrust::plus<int> binary_op;
    thrust::transform_exclusive_scan(thrust::host, A, A+N, A, unary_op, 4, binary_op); // in-place scan
    for(int i=0;i<N;i++){
        if(A[i]!=ans[i]){
            printf("test_5 run failed\n");
            exit(-1);
        }
    }

  printf("test_5 run passed!\n");
}

void test_6() { //raw ptr

   const int N=6;
    int A[N]={1, 0, 2, 2, 1, 3};
    int ans[N]={4, 3, 3, 1, -1, -2};
   
    thrust::negate<int> unary_op;
    thrust::plus<int> binary_op;
    thrust::transform_exclusive_scan( A, A+N, A, unary_op, 4, binary_op); // in-place scan
    for(int i=0;i<N;i++){
        if(A[i]!=ans[i]){
            printf("test_6 run failed\n");
            exit(-1);
        }
    }

  printf("test_6 run passed!\n");
}



int main(){
    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();

    return 0;
}