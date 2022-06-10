// ====------ user_defined_rules.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<iostream>
#include<cmath>
#include<functional>
#define CALL(x) x
#define CALL2(x) x

#if !defined(_MSC_VER)
#define __my_inline__ __forceinline
#else
#define __my_inline__ __inline__ __attribute__((always_inline))
#endif

#define VECTOR int
__forceinline__ __global__ void foo(){
  VECTOR a;
}

extern void foo3(std::function<int(int)> f);
extern int my_min(int a, int b);
void goo(std::function<int(int)> f) { f(0); };

int main(){
  int **ptr;
  cudaMalloc(ptr, 50);
  CALL(0);
  return 0;
}

class ClassA{
public:
  int fieldA;
  int fieldC;
  int methodA(int i, int j){return 0;};
};
class ClassB{
  int a;
public:
  int fieldB;
  int fieldD;
  int methodB(int i){return 0;}
  void set_a(int i){a = i;}
  int get_a(){return a;}
};

enum Fruit{
  apple,
  banana,
  pineapple
};

void foo2(){
  int c = 10;
  int d = 1;
  //CHECK: goo([&](int x) -> int {
  //CHECK-NEXT:   int m = std::min(x, 10);
  //CHECK-NEXT:   int n = std::min(x, 100), p = std::min(std::min(x, 10), 100);
  //CHECK-NEXT:   return std::min(c, d);
  //CHECK-NEXT: });
  foo3([&](int x)->int {
      int m = my_min(x, 10);
      int n = my_min(x, 100), p = my_min(my_min(x, 10), 100);
      return my_min(c, d);
  });
  //CHECK: CALL2(0);
  CALL(0);
  //CHECK: mytype *cu_st;
  CUstream_st *cu_st;

  //CHECK: ClassB a;
  //CHECK-NEXT: a.fieldD = 3;
  //CHECK-NEXT: a.methodB(2);
  //CHECK-NEXT: a.set_a(3);
  //CHECK-NEXT: int k = a.get_a();
  //CHECK-NEXT: Fruit f = pineapple;
  ClassA a;
  a.fieldC = 3;
  a.methodA(1,2);
  a.fieldA = 3;
  int k = a.fieldA;
  Fruit f = Fruit::apple;
}
