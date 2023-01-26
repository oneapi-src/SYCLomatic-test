#include "cudnn.h"
#include <iostream>

int main() {
  cudnnStatus_t status;

  const char *msg=cudnnGetErrorString(status);

  if (!msg)
    return 1;
    
  std::cout << "string = " << msg << "\n";

  return 0;
}