// ====------ onedpl_test_device_ptr.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <CL/sycl.hpp>

#include <iostream>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    else {
        std::cout << "PASS: " << msg << std::endl;
        return 0;
    }
}

int test_device_ptr_manipulation(void)
{
    int failing_tests = 0;
    typedef int T;

#ifdef DPCT_USM_LEVEL_NONE
    cl::sycl::buffer<T, 1> data(cl::sycl::range<1>(5));

    dpct::device_pointer<int> begin(data, 0);
    dpct::device_pointer<int> end(data, 5);
#else
    dpct::device_pointer<int> data(5);
    dpct::device_pointer<int> begin(data);
    dpct::device_pointer<int> end(data + 5);
#endif

    failing_tests += ASSERT_EQUAL("device_ptr test 1", end - begin, 5);

    begin++;
    begin--;

    failing_tests += ASSERT_EQUAL("device_ptr test 2", end - begin, 5);

    begin += 1;
    begin -= 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 3", end - begin, 5);

    begin = begin + (int) 1;
    begin = begin - (int) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 4", end - begin, 5);

    begin = begin + (unsigned int) 1;
    begin = begin - (unsigned int) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 5", end - begin, 5);

    begin = begin + (size_t) 1;
    begin = begin - (size_t) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 6", end - begin, 5);

    begin = begin + (ptrdiff_t) 1;
    begin = begin - (ptrdiff_t) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 7", end - begin, 5);

    begin = begin + (dpct::device_pointer<int>::difference_type) 1;
    begin = begin - (dpct::device_pointer<int>::difference_type) 1;

    failing_tests += ASSERT_EQUAL("device_ptr test 8", end - begin, 5);

    return failing_tests;
}

void test_device_ptr_iteration(void)
{
    typedef size_t T;

#ifdef DPCT_USM_LEVEL_NONE
    cl::sycl::buffer<T, 1> data(cl::sycl::range<1>(1024));

    dpct::device_pointer<T> begin(data, 0);
    dpct::device_pointer<T> end(data, 1024);
#else
    dpct::device_pointer<T> data(1024*sizeof(T));
    dpct::device_pointer<T> begin(data);
    dpct::device_pointer<T> end(data + 1024);
#endif
    auto policy = oneapi::dpl::execution::make_device_policy(dpct::get_default_queue());

    std::fill(policy, begin, end, 99);
    T result = oneapi::dpl::transform_reduce(policy, begin, end, static_cast<T>(0), std::plus<T>(), oneapi::dpl::identity());
    std::cout << "iteration result = " << result << ", expected = " << 99 * 1024 << "\n";
}

int main() {
    // FPGA device selector:  Emulator or Hardware
#ifdef FPGA_EMULATOR
    cl::sycl::intel::fpga_emulator_selector device_selector;
#elif defined(FPGA)
    cl::sycl::intel::fpga_selector device_selector;
#else
    // Initializing the devices queue with the default selector
    // The device queue is used to enqueue the kernels and encapsulates
    // all the states needed for execution
    cl::sycl::default_selector device_selector;
#endif

    std::unique_ptr<cl::sycl::queue> device_queue;
    try {
        device_queue.reset( new cl::sycl::queue(device_selector) );
    } catch (cl::sycl::exception const& e) {
        std::cout << "Caught a synchronous SYCL exception:" << std::endl
                  << e.what() << std::endl;
        std::cout << "If you are targeting an FPGA hardware, please ensure that your system is "
                     "plugged to an FPGA board that is set up correctly and compile with -DFPGA"
                  << std::endl;
        std::cout << "If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR."
                  << std::endl;
    }

    std::cout << "Device: " << device_queue->get_device().get_info<cl::sycl::info::device::name>()
              << std::endl;
    int failed_tests = test_device_ptr_manipulation();
    test_device_ptr_iteration();

    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
