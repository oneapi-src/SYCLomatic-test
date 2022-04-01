// ====------ testUnio.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include<iostream>
#include<cuda_runtime_api.h>
 


 class CZString {
  public:
   
   
  CZString(int aindex) {
  
  index_=aindex;
}

    struct StringStorage {
      unsigned policy_: 2;
      unsigned length_: 30; // 1GB max
    };

   
    union {
      int index_;
      StringStorage storage_;
    };
  };


int main()
{
   CZString s(2);
   std::cout<<"index_="<<s.index_<<std::endl;

return 0;

}


