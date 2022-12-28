#include "test.cuh"
#include "test.h"

int main() {
  f<<<1, 1>>>();
  return 0;
}
