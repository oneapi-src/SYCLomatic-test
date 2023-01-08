#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

int main(void)
{
    // H has storage for 4 integers
    std::vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;

    // Copy host_vector H to device_vector D
    dpct::device_vector<int> D = H;

    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;
    
    H = D;
    if (H[0] == 99 && H[1] == 88)
      return 0;
    else
      return 1;
}