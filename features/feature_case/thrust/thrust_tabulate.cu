// ====------ tabulate.cu------------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-----------------------------------------------------------------===//


#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

void test_1() {
    const int N=10;
    int A[N];
    int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
    thrust::host_vector<int> V(A,A+N);
   
    thrust::tabulate(thrust::host, V.begin(), V.end(), thrust::negate<int>());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_1 run failed\n");
            exit(-1);
        }
    }

  printf("test_1 run passed!\n");
}

void test_2() {

  const int N=10;
    int A[N];
    int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
    thrust::host_vector<int> V(A,A+N);
   
    thrust::tabulate( V.begin(), V.end(), thrust::negate<int>());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_2 run failed\n");
            exit(-1);
        }
    }

  printf("test_2 run passed!\n");
}


void test_3() {
    const int N=10;
    int A[N];
    int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
    thrust::device_vector<int> V(A,A+N);
   
    thrust::tabulate(thrust::device, V.begin(), V.end(), thrust::negate<int>());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_3 run failed\n");
            exit(-1);
        }
    }

  printf("test_3 run passed!\n");
}

void test_4() {

 const int N=10;
    int A[N];
    int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
    thrust::device_vector<int> V(A,A+N);
   
    thrust::tabulate(V.begin(), V.end(), thrust::negate<int>());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_4 run failed\n");
            exit(-1);
        }
    }

  printf("test_4 run passed!\n");
}


void test_5() {
    const int N=10;
    int A[N];
    int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
    
    thrust::tabulate(thrust::host, A,A+N, thrust::negate<int>());
    for(int i=0;i<N;i++){
        if(A[i]!=ans[i]){
            printf("test_5 run failed\n");
            exit(-1);
        }
    }

  printf("test_5 run passed!\n");
}

void test_6() {

  const int N=10;
    int A[N];
    int ans[N]={0, -1, -2, -3, -4, -5, -6, -7, -8, -9};
    
    thrust::tabulate( A,A+N, thrust::negate<int>());
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
