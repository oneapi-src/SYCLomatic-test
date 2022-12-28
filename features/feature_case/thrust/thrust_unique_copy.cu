// ====------ thrust_unique_copy.cu------------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------------===//

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


void test_1() {  // host iterator
    const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::host_vector<int> V(A,A+N);
    thrust::host_vector<int> result(B,B+M);
   
    thrust::host_vector<int>::iterator result_end = thrust::unique_copy(thrust::host, V.begin(), V.end(), result.begin());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_1 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_1 run failed\n");
        exit(-1);
    }

  printf("test_1 run passed!\n");
}

void test_2() { // host iterator

  const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::host_vector<int> V(A,A+N);
    thrust::host_vector<int> result(B,B+M);
   
    thrust::host_vector<int>::iterator result_end = thrust::unique_copy( V.begin(), V.end(), result.begin());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_2 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_2 run failed\n");
        exit(-1);
    }

  printf("test_2 run passed!\n");
}


void test_3() { // host iterator
    const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::host_vector<int> V(A,A+N);
    thrust::host_vector<int> result(B,B+M);
   
    thrust::host_vector<int>::iterator result_end = thrust::unique_copy(thrust::host, V.begin(), V.end(), result.begin(), thrust::equal_to<int>());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_3 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_3 run failed\n");
        exit(-1);
    }

  printf("test_3 run passed!\n");
}

void test_4() { // host iterator

  const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::host_vector<int> V(A,A+N);
    thrust::host_vector<int> result(B,B+M);
   
    thrust::host_vector<int>::iterator result_end = thrust::unique_copy( V.begin(), V.end(), result.begin(), thrust::equal_to<int>());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_4 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_4 run failed\n");
        exit(-1);
    }

  printf("test_4 run passed!\n");
}




void test_5() { // device iterator
    const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::device_vector<int> V(A,A+N);
    thrust::device_vector<int> result(B,B+M);
   
    thrust::device_vector<int>::iterator result_end = thrust::unique_copy(thrust::device, V.begin(), V.end(), result.begin());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_5 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_5 run failed\n");
        exit(-1);
    }

  printf("test_5 run passed!\n");
}

void test_6() { // device iterator

  const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::device_vector<int> V(A,A+N);
    thrust::device_vector<int> result(B,B+M);
   
    thrust::device_vector<int>::iterator result_end = thrust::unique_copy( V.begin(), V.end(), result.begin());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_6 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_6 run failed\n");
        exit(-1);
    }

  printf("test_6 run passed!\n");
}


void test_7() { // device iterator
    const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::device_vector<int> V(A,A+N);
    thrust::device_vector<int> result(B,B+M);
   
    thrust::device_vector<int>::iterator result_end = thrust::unique_copy(thrust::device, V.begin(), V.end(), result.begin(), thrust::equal_to<int>());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_7 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_7 run failed\n");
        exit(-1);
    }

  printf("test_7 run passed!\n");
}

void test_8() { // device iterator

 const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
    thrust::device_vector<int> V(A,A+N);
    thrust::device_vector<int> result(B,B+M);
   
    thrust::device_vector<int>::iterator result_end = thrust::unique_copy( V.begin(), V.end(), result.begin(), thrust::equal_to<int>());
    for(int i=0;i<M;i++){
        if(result[i]!=ans[i]){
            printf("test_8 run failed\n");
            exit(-1);
        }
    }
    if(result_end-result.begin()!=4){
        printf("test_8 run failed\n");
        exit(-1);
    }

  printf("test_8 run passed!\n");
}



void test_9() { // ptr iterator
    const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
   
    thrust::unique_copy(thrust::host,A, A + N, B);
    for(int i=0;i<M;i++){
        if(B[i]!=ans[i]){
            printf("test_9 run failed\n");
            exit(-1);
        }
    }

  printf("test_9 run passed!\n");
}

void test_10() { // ptr iterator

  const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
   
    thrust::unique_copy(A, A + N, B);
    for(int i=0;i<M;i++){
        if(B[i]!=ans[i]){
            printf("test_10 run failed\n");
            exit(-1);
        }
    }

  printf("test_10 run passed!\n");
}


void test_11() { // ptr iterator
     const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
   
    thrust::unique_copy(thrust::host,A, A + N, B, thrust::equal_to<int>());
    for(int i=0;i<M;i++){
        if(B[i]!=ans[i]){
            printf("test_11 run failed\n");
            exit(-1);
        }
    }

  printf("test_11 run passed!\n");
}

void test_12() { // ptr iterator

   const int N=7;
    int A[N]={1, 3, 3, 3, 2, 2, 1};
    int B[N];
    const int M=N-3;
    int ans[M]={1, 3, 2, 1};
   
    thrust::unique_copy(A, A + N, B, thrust::equal_to<int>());
    for(int i=0;i<M;i++){
        if(B[i]!=ans[i]){
            printf("test_12 run failed\n");
            exit(-1);
        }
    }

  printf("test_12 run passed!\n");
}




int main(){
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