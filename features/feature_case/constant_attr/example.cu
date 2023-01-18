#include <stdio.h>

void init_l1(int *hvals); void get_l1(int *target); // from no_header1.cu
void init_l2(int *hvals); void get_l2(int *target); // from no_header2.cu
void init_h1(int *hvals); void get_h1(int *target); // from use_header1.cu
void init_h2(int *hvals); void get_h2(int *target); // from use_header2.cu

int main() {
  int hvals[2];
  int target[2];

  // Initialize __constant__ int dvals[] array declared in each [no|use]_header[1|2].cu file
  hvals[0] =  123;  hvals[1] =  357; init_l1(hvals); 
  hvals[0] = 1123;  hvals[1] = 1357; init_l2(hvals); 
  hvals[0] = 2123;  hvals[1] = 2357; init_h1(hvals);
  hvals[0] = 3123;  hvals[1] = 3357; init_h2(hvals);     

  // Verify dvals[] arrays are distinct by checking values
  get_l1(target); if (target[0] !=  123 || target[1] !=  357) return 1;
  get_l2(target); if (target[0] != 1123 || target[1] != 1357) return 1;
  get_h1(target); if (target[0] != 2123 || target[1] != 2357) return 1;
  get_h2(target); if (target[0] != 3123 || target[1] != 3357) return 1;

  fprintf(stderr,"Passed\n");
  return 0;
}