#ifndef __XPUSTREAM_H__
#define __XPUSTREAM_H__

#include <sycl/sycl.hpp>

namespace c10 {
namespace xpu {
class XPUStream {
public:
  XPUStream() { _q_ptr = &_q; }
  sycl::queue &queue() const { return *_q_ptr; }
  operator sycl::queue &() const { return *_q_ptr; }
  operator sycl::queue *() const { return _q_ptr; }

private:
  sycl::queue *_q_ptr;
  sycl::queue _q;
};
XPUStream getCurrentXPUStream() {
  static XPUStream stream;
  return stream;
}
} // namespace xpu
} // namespace c10

#endif // __XPUSTREAM_H__
