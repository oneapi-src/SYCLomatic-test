#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "shared.hpp"

#define PTXFILE    "premade.ptx"

static void loadModule(CUmodule &module, const char *ptxfile, int load_type) {
if (load_type==0) {
  checkErrors( cuModuleLoad(&module,ptxfile) );
} else if (load_type==1 || load_type==2) {
  std::ifstream      ifile(ptxfile, std::ios::in | std::ios::binary);
  std::stringstream  strStream; 
  std::string        content;

  // put file into a buffer for loading
  strStream << ifile.rdbuf();
  content = strStream.str();
  
  if (load_type==1) {
    checkErrors( cuModuleLoadData(&module,content.c_str()) );
  } else {
#ifdef TODO_LOADDATAEX_OPTIONS
    const unsigned int jitNumOptions = 1;
    CUjit_option  jitOptions[jitNumOptions];
    void         *jitOptVals[jitNumOptions];
  
    // set up wall clock time
    jitOptions[0] = CU_JIT_WALL_TIME;
#endif

    checkErrors( cuModuleLoadDataEx(&module,content.c_str(),
#ifdef TODO_LOADDATAEX_OPTIONS
                                    jitNumOptions, jitOptions, (void **)jitOptVals)
#endif
                                    0, NULL, NULL)
                );

#ifdef TODO_LOADDATAEX_OPTIONS
    unsigned long long opt_int = (unsigned long long) jitOptVals[0];
    float jitTime = *((float *) &opt_int);
      
    std::cout << "\tcompile-time: " << jitTime << "ms\n";
#endif
  }
}
 
}

int run_tests() {
  static int cnt;
  int load_type;
  int invoke_type;

  for (load_type=0; load_type<3; load_type++) {
    for (invoke_type=0; invoke_type<2; invoke_type++) {
      int seed = SEED;

      // allocate memory
      int *a;
      int *b;
      int *c;
  
      cudaMallocManaged((void **) &a, sizeof(int) * VEC_LENGTH);
      cudaMallocManaged((void **) &b, sizeof(int) * VEC_LENGTH);
      cudaMallocManaged((void **) &c, sizeof(int) * VEC_LENGTH);

      for (int i=0; i<VEC_LENGTH; i++) {
        b[i] = i   + load_type*53 + invoke_type*97 + cnt*101;
        c[i] = i+1 + load_type*53 + invoke_type*97 + cnt*101;
      }

      // test ptx module-loading, kernel function invocation, and verify results

      // load_type==0 use cuModuleLoad
      // load_type==1 use cuModuleLoadData
      // load_type==2 use cuModuleLoadDataEx 

      // load pre-made ptx
      CUmodule   module;
      loadModule(module,PTXFILE,load_type);

      if (module==NULL) {
        std::cout << "No module\n";
        exit(-1);
      }

      // find kernel function 
      CUfunction function;  
      checkErrors( cuModuleGetFunction(&function, module, "foo") );

      if (function==NULL) {
        std::cout << "No function\n";
        exit(-1);
      }
    
      // test cuFuncGetAttribute
      {
        int size;
        CUresult result = cuFuncGetAttribute(&size, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
        if (result != CUDA_SUCCESS) {
          std::cout << "cuFuncGetAttribute failed\n";
          return 1;
        }
      }
    
      // invoke kernel function
      if (invoke_type==0) {
        void *args[3] = { &a, &b, &c };
        checkErrors( cuLaunchKernel(function,
                                    VEC_LENGTH, 1, 1,  // 1x1x1 blocks
                                    1,          1, 1,  // 1x1x1 threads
                                    0, 0, args, 0) );
      } else if (invoke_type==1) {
        int    *argBuffer[3] = { a, b, c };
        size_t  argBufferSize = sizeof(argBuffer);
        void   *extra[] = {
#ifdef TODO_EXTRA_BUFFER_ANNOTATIONS
          CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,        
          CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
          CU_LAUNCH_PARAM_END   
#else
          ((void *) 2), &argBufferSize,        
          ((void *) 1),  argBuffer,
          ((void *) 0)
#endif
        };
        
        checkErrors( cuLaunchKernel(function,
                                    VEC_LENGTH, 1, 1,  // 1x1x1 blocks
                                    1,          1, 1,  // 1x1x1 threads
                                    0, 0, 0, extra) );
      }

      cudaDeviceSynchronize();

      // check result
      for (int i=0; i<VEC_LENGTH; i++) {
        if (a[i]!=(b[i]*c[i]+seed)) {
          printf("\tRESULT: Failed a[%d]=%d, b[%d]=%d, c[%d]=%d %d\n",
                 i,a[i],
                 i,b[i],
                 i,c[i],
                 seed);
          return 1;
        }
      }
      ++cnt;
      if (cnt&0x3) {
        // Do not unload every time, so that we exercise automatic unloading on program exit with Windows
        cuModuleUnload(module);
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  for (auto i=0; i<5; i++) {
    if (run_tests())
      return 1;
  }
  return 0;
}
