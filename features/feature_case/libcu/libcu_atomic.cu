#include <cuda/atomic>

int main(){
  cuda::atomic<int> a;
  cuda::atomic<int> b(0);
  cuda::atomic<int, cuda::thread_scope_block> c(0);
  int ans = c.load();
  a.store(0);
  int ans2 = a.load();

  int tmp =1,tmp1=2;
  ans = a.exchange(1);
  a.compare_exchange_weak(tmp,2);
  a.compare_exchange_strong(tmp1,3);

  ans = a.fetch_add(1);
  ans = a.fetch_sub(-1);
  ans = a.fetch_and(2);
  ans = a.fetch_or(1);
  ans = a.fetch_xor(2);
  return 0;
}