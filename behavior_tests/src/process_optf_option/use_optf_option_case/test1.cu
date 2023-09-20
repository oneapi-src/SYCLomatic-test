#include <cuda.h>


#ifdef __FOO1__
__global__ void foo1() {}
#endif

#ifdef __FOO2__
__global__ void foo2() {}
#endif

#ifdef __FOO3__
__global__ void foo3() {}
#endif

#ifdef __FOO4__
__global__ void foo4() {}
#endif



#include "foo1.h"
#include "foo2.h"
#include "foo3.h"
#include "foo4.h"


int main() {

 
  return 0;
}
