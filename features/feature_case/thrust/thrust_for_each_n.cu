#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


struct add_functor
{
  __host__ __device__
  void operator()(int & x)
  {
    x++;
  }
};


void test_1() {
    const int N=3;
    int A[N]={0,1,2};
    int ans[N]={1,2,3};
    thrust::host_vector<int> V(A,A+N);
   
    thrust::for_each_n(thrust::host, V.begin(), V.size(), add_functor());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_1 run failed\n");
            exit(-1);
        }
    }

  printf("test_1 run passed!\n");
}

void test_2() {

 const int N=3;
    int A[N]={0,1,2};
    int ans[N]={1,2,3};
    thrust::host_vector<int> V(A,A+N);
   
    thrust::for_each_n( V.begin(), V.size(), add_functor());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_2 run failed\n");
            exit(-1);
        }
    }

  printf("test_2 run passed!\n");
}


void test_3() {
    const int N=3;
    int A[N]={0,1,2};
    int ans[N]={1,2,3};
    thrust::device_vector<int> V(A,A+N);
   
    thrust::for_each_n(thrust::device, V.begin(), V.size(), add_functor());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_3 run failed\n");
            exit(-1);
        }
    }

  printf("test_3 run passed!\n");
}

void test_4() {

  const int N=3;
    int A[N]={0,1,2};
    int ans[N]={1,2,3};
    thrust::device_vector<int> V(A,A+N);
   
    thrust::for_each_n( V.begin(), V.size(), add_functor());
    for(int i=0;i<N;i++){
        if(V[i]!=ans[i]){
            printf("test_4 run failed\n");
            exit(-1);
        }
    }

  printf("test_4 run passed!\n");
}


void test_5() {
    const int N=3;
    int A[N]={0,1,2};
    int ans[N]={1,2,3};
    
    thrust::for_each_n(thrust::host, A, N, add_functor());
    for(int i=0;i<N;i++){
        if(A[i]!=ans[i]){
            printf("test_5 run failed\n");
            exit(-1);
        }
    }

  printf("test_5 run passed!\n");
}

void test_6() {

  const int N=3;
    int A[N]={0,1,2};
    int ans[N]={1,2,3};
    
    thrust::for_each_n( A, N, add_functor());
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
