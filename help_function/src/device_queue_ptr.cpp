#include <iostream>
#include <dpct/dpct.hpp>

void foo(int n) {
  ((dpct::queue_ptr) n)->wait();
}

struct queue_wrapper {
  dpct::queue_ptr p;
  queue_wrapper() : p(0) {};
};

int main() {
  int res = 0;
  auto check = [&res](std::string msg, bool b) {
    if (!b) {
      res = 1;
      std::cout << msg << " fail\n";
    }
  };

  auto& q = dpct::get_default_queue();
  auto is_default = [&q](auto p) { return &q == &*p; };

  dpct::queue_ptr p1(0);
  dpct::queue_ptr p2(nullptr);
  int x = p1;
  dpct::queue_ptr p3(x);

  dpct::queue_ptr arr[] = {0, 0U, 0L, 0UL, 0LL, 0ULL};

  check("queue_ptr(0) is default", is_default(p1));
  check("queue_ptr(nullptr) is default", is_default(p2));
  check("queue_ptr(x) is default", is_default(p3));
  check("queue_wrapper is default", is_default(queue_wrapper().p));

  foo(0);

  return res;
}
