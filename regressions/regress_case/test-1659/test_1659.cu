/*
* This test case is used to detect the issue that ", item_ct1" is inserted into
* the invalid place of migrated code.
 */

#include <stdio.h>

namespace utils {
static __device__ __forceinline__ float bar( float r, int mask, int warp_size )
{
  int i = blockIdx.x;
  return 0.0f;
}

template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
struct foo
{
  template< typename Operator, typename Value_type >
  static __device__ __inline__ Value_type execute( Value_type x )
  {
    for( int mask = WARP_SIZE / 2 ; mask >= NUM_THREADS_PER_ITEM ; mask >>= 1 )
      x = Operator::eval( x, bar(x, mask) );
    return x;
  }
};

template< int NUM_THREADS_PER_ITEM, int WARP_SIZE = 32 >
struct Warp_reduce : public foo<NUM_THREADS_PER_ITEM, WARP_SIZE> {};

}

int main(){
  return 0;
}
