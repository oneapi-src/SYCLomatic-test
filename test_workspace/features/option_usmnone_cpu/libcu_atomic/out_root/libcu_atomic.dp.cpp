#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/atomic.hpp>

int main(){
  dpct::atomic<int> a;
  dpct::atomic<int> b(0);
  dpct::atomic<int, sycl::memory_scope::work_group, sycl::memory_order::relaxed>
      c(0);
  int ans = c.load();
  a.store(0);
  int ans2 = a.load();

  int tmp =1,tmp1=2;
  ans = a.exchange(1);
  a.compare_exchange_weak(tmp,2);
  a.compare_exchange_strong(tmp1,3);

  ans = a.fetch_add(1);
  ans = a.fetch_sub(-1);
  return 0;
}