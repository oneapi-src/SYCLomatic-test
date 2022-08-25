#include <iostream>
#include <math.h>

__global__
void recip( double *x, double *y)
{
    int a = *x;

    y[0] = __drcp_rd(*x);
    y[1] = __drcp_rn(*x);
    y[2] = __drcp_ru(*x);
    y[3] = __drcp_rz(*x);
}

static bool moreThanOneLSBDiff(double a, double b) {
  union dbl_and_ll {
    double     dbl;
    long long   ll;
  };

  dbl_and_ll cmp_a, cmp_b;

  cmp_a.dbl = a;
  cmp_b.dbl = b;

  if (abs(cmp_a.ll-cmp_b.ll)>1) {
    return true;
  }
  return false;
}

int main(void)
{
  double *x, *y;

  cudaMallocManaged(&x, sizeof(double));
  cudaMallocManaged(&y, 4*sizeof(double));

  *x = 1.234567891234;
 
  for (int i=0; i<10; i++)  {
    std::cout << "Iteration " << i << " " << 1.0/x[0] << "\n";
    recip<<<1, 1>>>(x, y);

   cudaDeviceSynchronize();

   // ensure that migrated drcp* intrinsics produce results that are at
   // most one-bit off (due to rounding) of standard double-precision division.  
   // This ensures that migrated code still has double-precision accuracy.
   if (moreThanOneLSBDiff(y[0],(1.0/x[0])) ||
       moreThanOneLSBDiff(y[1],(1.0/x[0])) ||
       moreThanOneLSBDiff(y[2],(1.0/x[0])) ||
       moreThanOneLSBDiff(y[3],(1.0/x[0]))) {
      std::cout << "Failed\n" ;
      return 1;
    } 
    (*x)+=123.1241241;
  }

  cudaFree(x);
  cudaFree(y);
  
  std::cout << "Passed\n";
  return 0;
}