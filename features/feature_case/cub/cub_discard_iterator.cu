#include <cub/cub.cuh>

bool test() {
  cub::DiscardOutputIterator<> Iter, Begin = Iter;
  for (int i = 0; i < 10; ++i, Iter++) {
    *Iter = i;
  }

  return Iter - Begin == 10;
}

int main() {
  if (!test())
    return 1;
  return 0;
}
