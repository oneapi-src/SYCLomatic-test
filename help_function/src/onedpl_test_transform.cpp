// ====------ onedpl_test_transform.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"
#include "oneapi/dpl/algorithm"

#include "dpct/dpct.hpp"
#include "dpct/dpl_utils.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

template<typename String, typename _T1, typename _T2>
int ASSERT_EQUAL(String msg, _T1&& X, _T2&& Y) {
    if(X!=Y) {
        std::cout << "FAIL: " << msg << " - (" << X << "," << Y << ")" << std::endl;
        return 1;
    }
    return 0;
}

int test_passed(int failing_elems, std::string test_name) {
    if (failing_elems == 0) {
        std::cout << "PASS: " << test_name << std::endl;
        return 0;
    }
    return 1;
}

template<typename Buffer> void fill_buffer(Buffer& src_buf, int start_index, int end_index, uint64_t value) {
    auto src = src_buf.template get_access<sycl::access::mode::write>();
    for (int i = start_index; i != end_index; ++i) {
        src[i] = value;
    }
}

template<typename Buffer> void iota_buffer(Buffer& dst_buf, int start_index, int end_index, int offset) {
    auto dst = dst_buf.template get_access<sycl::access::mode::write>();
    for (int i = start_index; i != end_index; ++i) {
        dst[i] = i + offset;
    }
}

template<typename Buffer> void iota_reverse_buffer(Buffer& dst_buf, int start_index, int end_index, int offset) {
    auto dst = dst_buf.template get_access<sycl::access::mode::write>();
    for (int i = start_index; i != end_index; ++i) {
        dst[i] = (end_index - start_index - 1) - i + offset;
    }
}

template<typename Array> void fill_array(Array &arr, int start_index, int end_index, int value) {
    for (int i = start_index; i != end_index; ++i) {
        arr[i] = value;
    }
}

template<typename Array> void iota_array(Array &arr, int start_index, int end_index, int offset) {
    for (int i = start_index; i != end_index; ++i) {
        arr[i] = i + offset;
    }
}

template<typename Array> void iota_reverse_array(Array &arr, int start_index, int end_index, int offset) {
    for (int i = start_index; i != end_index; ++i) {
        arr[i] = (end_index - start_index - 1) - i + offset;
    }
}

template<typename Queue, typename Array> void host_to_device(Queue &q, Array hostArray, Array &deviceArray) {
    q.submit([&](sycl::handler& h) {
        h.memcpy(deviceArray, hostArray, 8 * sizeof(int));
    });
    q.wait();
}

template<typename Queue, typename Array> void device_to_host(Queue &q, Array hostArray, Array &deviceArray) {
    q.submit([&](sycl::handler& h) {
        h.memcpy(hostArray, deviceArray, 8 * sizeof(int));
    });
    q.wait();
}

struct add_tuple_components {
    template <typename Scalar, typename Tuple>
    Scalar operator()(const Scalar& s, const Tuple& t) const {
        return s + std::get<0>(t) + std::get<1>(t) +std::get<2>(t);
    }
};

struct add_tuple_components2 {
    template <typename Scalar, typename Tuple>
    Scalar operator()(const Scalar& s, const Tuple& t) const {
        return s +std::get<0>(t) +std::get<1>(t);
    }
};

struct add_tuple_components3 {
    template <typename Scalar, typename Tuple>
    Scalar operator()(const Scalar s, const Tuple t) const {
        return s +std::get<0>(t) +std::get<1>(t) +std::get<2>(t) +std::get<3>(t) +std::get<4>(t);
    }
};

struct add_tuple_components4 {
    template <typename Scalar>
    int operator()(const Scalar s, const Scalar t) const {
        return s + t;
    }
};

struct add_to_tuple_components {
    template <typename Element, typename TupleElement>
    typename std::decay<TupleElement>::type
    operator()(const Element& e, const TupleElement& t) const {
        typedef typename std::decay<TupleElement>::type result_type;

        return result_type(e + std::get<0>(t), e + std::get<1>(t));
    }
};

template<typename Result>
struct add_to_tuple_components2 {
    template <typename Element, typename Element2>
    Result operator()(const Element& e, const Element2& t) const {

        return Result(e + t, e + t);
    }
};

struct add_to_tuple_components3 {
    template <typename TupleElement, typename TupleElement2>
    typename std::decay<TupleElement>::type
    operator()(TupleElement && e, TupleElement2&& t) const {
        typedef typename std::decay<TupleElement>::type result_type;

        return result_type(std::get<0>(e) +std::get<0>(t) +std::get<1>(t),std::get<1>(e) +std::get<2>(t) +std::get<3>(t));
    }
};

struct negate_tuple_components {
    template <typename Tuple>
    int operator()(const Tuple t) const {
        return -std::get<0>(t);
    }
};

struct square {
    int operator()(int x) const {
        return x*x;
    }
};

struct assign_square {
    template<typename T>
    void operator()(T&& t) const {
       std::get<1>(t) =std::get<0>(t) *std::get<0>(t);
    }
};

// added functions to make test USM compatible (with device_vector or device_pointer)

// transform(buffer, buffer, buffer, buffer)
template <typename Policy, typename Iterator1, typename Iterator2, typename IteratorResult>
void transform_call(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 first2, IteratorResult result) {
    std::transform(policy, first1, last1, first2, result, std::plus<int>());
}

// transform(buffer, buffer, buffer)
template <typename Policy, typename Iterator1, typename IteratorResult>
void transform_call(Policy policy, Iterator1 first1, Iterator1 last1, IteratorResult result) {
    std::transform(policy, first1, last1, result, std::negate<int>());
}

// transform(buffer, buffer, perm_it, buffer)
template <typename Policy, typename Iterator1, typename Iterator2, typename IteratorResult>
void transform_call2(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 perm_input1, Iterator2 perm_input2, IteratorResult result) {
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_permutation_iterator(perm_input1, perm_input2),
        result,
        std::plus<int>()
    );
}

// transform(buffer, buffer, zip_it<buffer x3>, buffer)
template <typename Policy, typename Iterator1, typename Iterator2, typename IteratorResult>
void transform_call3(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 zip_input1, Iterator2 zip_input2, Iterator2 zip_input3, IteratorResult result) {
// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_zip_iterator(zip_input1, zip_input2, zip_input3),
        result,
        add_tuple_components()
    );
#endif
}

// transform(buffer, buffer, transform_it<buffer, func>, buffer)
template <typename Policy, typename Iterator1, typename Iterator2, typename Functor, typename IteratorResult>
void transform_call4(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 trf_input, Functor func, IteratorResult result) {
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_transform_iterator(trf_input, func),
        result,
        std::plus<int>()
    );
}

// transform(buffer, buffer, buffer, perm_it)
template <typename Policy, typename Iterator1, typename Iterator2, typename IteratorResult>
void transform_call5(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 first2, IteratorResult perm_input1, IteratorResult perm_input2) {
    std::transform
    (
        policy, first1, last1, first2,
        oneapi::dpl::make_permutation_iterator(perm_input1, perm_input2),
        std::plus<int>()
    );
}

// transform(buffer, buffer, buffer, zip_it<perm_it, perm_it>)
template <typename Policy, typename Iterator1, typename Iterator2, typename IteratorResult>
void transform_call6(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 first2, IteratorResult perm_input1, IteratorResult perm_input2, IteratorResult perm_map_input) {
    auto tmp = oneapi::dpl::make_zip_iterator(perm_input1, perm_input2);

    typedef typename ::std::iterator_traits<decltype(tmp)>::value_type TupleT;
    std::transform
    (
        policy, first1, last1, first2,
        oneapi::dpl::make_zip_iterator
        (
            perm_input1, perm_input2
//            oneapi::dpl::make_permutation_iterator(perm_input1, perm_map_input),
//            oneapi::dpl::make_permutation_iterator(perm_input2, perm_map_input)
        ),
        add_to_tuple_components2<TupleT>()
    );
}

// transform(buffer, buffer, zip_it<perm_it, perm_it>, perm_it)
template <typename Policy, typename Iterator1, typename Iterator2>
void transform_call7(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 perm_input1, Iterator2 perm_input2, Iterator2 perm_map_input) {
    auto perm1 = oneapi::dpl::make_permutation_iterator(perm_input1, perm_map_input);
    auto perm2 = oneapi::dpl::make_permutation_iterator(perm_input2, perm_map_input);
    auto zip = oneapi::dpl::make_zip_iterator(perm1, perm2);

// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_zip_iterator
        (
            oneapi::dpl::make_permutation_iterator(perm_input1, perm_map_input),
            oneapi::dpl::make_permutation_iterator(perm_input2, perm_map_input)
        ),
        oneapi::dpl::make_permutation_iterator(perm_input1, perm_map_input),
        add_tuple_components2()
    );
#endif
}

// transform(buffer, buffer, zip_it<perm_it, buffer x4>, buffer)
template <typename Policy, typename Iterator1>
void transform_call8(Policy policy, Iterator1 first1, Iterator1 last1, Iterator1 input2, Iterator1 input3, Iterator1 input4, Iterator1 input5, Iterator1 input6, Iterator1 map_input) {
// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_zip_iterator
        (
            oneapi::dpl::make_permutation_iterator(input2, map_input),
            input3,
            input4,
            input5,
            input6
        ),
        input3,
        add_tuple_components3()
    );
#endif
}

// transform(buffer, buffer, zip_it<buffer x2>, zip_it<buffer x2>)
template <typename Policy, typename Iterator1>
void transform_call9(Policy policy, Iterator1 first1, Iterator1 last1, Iterator1 input2, Iterator1 input3) {
// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_zip_iterator(input2, input3),
        oneapi::dpl::make_zip_iterator(input2, input3),
        add_to_tuple_components()
    );
#endif
}

// transform(perm_it, perm_it, buffer, perm_it)
template <typename Policy, typename Iterator1>
void transform_call10(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input_map) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;

    std::transform
    (
        policy,
        perm_begin,
        perm_end,
        input2,
        perm_begin,
        std::plus<int>());
}

// transform(perm_it<perm_it, buffer>, perm_it<perm_it, buffer>, perm_it, perm_it<perm_it, buffer>
template <typename Policy, typename Iterator1>
void transform_call11(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input_map, Iterator1 input_map2) {
    auto perm_inside_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_inside_end = perm_inside_begin + 4;

    auto perm_outside_begin = oneapi::dpl::make_permutation_iterator(perm_inside_begin, input2);
    auto perm_outside_end = perm_outside_begin + 4;

    std::transform
    (
        policy,
        perm_outside_begin,
        perm_outside_end,
        oneapi::dpl::make_permutation_iterator(input2, input_map2),
        perm_outside_begin,
        std::plus<int>()
    );
}

// transform(perm_it, perm_it, transform_it<zip_it<buffer x2>, func>, perm_it)
template <typename Policy, typename Iterator1>
void transform_call12(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input_map) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;

// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy,
        perm_begin,
        perm_end,
        oneapi::dpl::make_transform_iterator
        (
            oneapi::dpl::make_zip_iterator(input2, input2 + 1),
            assign_square()
        ),
        perm_begin,
        add_tuple_components2()
    );
#endif
}

// transform(perm_it, perm_it, zip_it<perm_it, perm_it>, perm_it)
template <typename Policy, typename Iterator1>
void transform_call13(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input3, Iterator1 input_map) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;

// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy,
        perm_begin,
        perm_end,
        oneapi::dpl::make_zip_iterator
        (
            oneapi::dpl::make_permutation_iterator(input2, input_map),
            oneapi::dpl::make_permutation_iterator(input3, input_map)
        ),
        perm_begin,
        add_tuple_components2()
    );
#endif
}

// transform(perm_it, perm_it, transform_it<perm_it, func>, perm_it)
template <typename Policy, typename Iterator1>
void transform_call14(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input_map) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;

    std::transform
    (
        policy,
        perm_begin,
        perm_end,
        oneapi::dpl::make_transform_iterator
        (
            oneapi::dpl::make_permutation_iterator(input2, input_map),
            square()
        ),
        perm_begin,
        std::plus<int>()
    );
}

// transform(perm_it, perm_it, perm_it)
template <typename Policy, typename Iterator1>
void transform_call15(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input_map) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;
    auto perm2_begin = oneapi::dpl::make_permutation_iterator(input2, input_map);

    std::transform
    (
        policy,
        perm_begin,
        perm_end,
        perm2_begin,
        std::negate<int>()
    );
}

// transform(perm_it, perm_it, buffer)
template <typename Policy, typename Iterator1>
void transform_call16(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input_map) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;

    std::transform
    (
        policy,
        perm_begin,
        perm_end,
        input2,
        std::negate<int>()
    );
}

// transform(perm_it, perm_it, perm_it, buffer)
template <typename Policy, typename Iterator1>
void transform_call17(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input_map, Iterator1 input_map2) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;
    auto perm2_begin = oneapi::dpl::make_permutation_iterator(input1, input_map2);

    std::transform
    (
        policy,
        perm_begin,
        perm_end,
        perm2_begin,
        input2,
        std::plus<int>()
    );
}

// transform(counting_it, counting_it, perm_it, perm_it)
template <typename Policy, typename Iterator1, typename Iterator2>
void transform_call18(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 input1, Iterator2 input_map) {
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_permutation_iterator(input1, input_map),
        oneapi::dpl::make_permutation_iterator(input1, input_map),
        std::plus<int>()
    );
}

// transform(counting_it, counting_it, zip_it<buffer x2>, buffer)
template <typename Policy, typename Iterator1, typename Iterator2, typename IteratorResult>
void transform_call19(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 zip_input1, Iterator2 zip_input2, IteratorResult result) {
// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_zip_iterator(zip_input1, zip_input2),
        result,
        add_tuple_components2()
    );
#endif
}

// transform(zip_it<buffer x2, perm_it x2>, zip_it<buffer x2, perm_it x2>, buffer)
template <typename Policy, typename Iterator1>
void transform_call20(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input3, Iterator1 input4, Iterator1 input_map, Iterator1 input_map2) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input3, input_map);
    auto perm_end = perm_begin + 4;
    auto perm_begin2 = oneapi::dpl::make_permutation_iterator(input3, input_map2);
    auto perm_end2 = perm_begin2 + 4;

    std::transform
    (
        policy,
        oneapi::dpl::make_zip_iterator
        (
            input1,
            perm_begin,
            input2,
            perm_begin2
        ),
        oneapi::dpl::make_zip_iterator
        (
            input1 + 4,
            perm_end,
            input2 + 4,
            perm_end2
        ),
        input4,
        negate_tuple_components()
    );
}

// transform(zip_it<perm_it x2>, zip_it<perm_it x2>, zip_it<perm_it x4>, zip_it<perm_it x2>)
template <typename Policy, typename Iterator1>
void transform_call21(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input3, Iterator1 input4, Iterator1 input5, Iterator1 input6, Iterator1 input_map) {
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input1, input_map);
    auto perm_end = perm_begin + 4;
    auto perm_begin2 = oneapi::dpl::make_permutation_iterator(input2, input_map);
    auto perm_end2 = perm_begin2 + 4;

// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy,
        oneapi::dpl::make_zip_iterator
        (
            perm_begin,
            perm_begin2
        ),
        oneapi::dpl::make_zip_iterator
        (
            perm_end,
            perm_end2
        ),
        oneapi::dpl::make_zip_iterator
        (
            oneapi::dpl::make_permutation_iterator(input3, input_map),
            oneapi::dpl::make_permutation_iterator(input4, input_map),
            oneapi::dpl::make_permutation_iterator(input5, input_map),
            oneapi::dpl::make_permutation_iterator(input6, input_map)
        ),
        oneapi::dpl::make_zip_iterator
        (
            perm_begin,
            perm_begin2
        ),
        add_to_tuple_components3()
    );
#endif
}

// transform(constant_it, constant_it, transform_it<perm_it, func>, perm_it)
template <typename Policy, typename Iterator1, typename Iterator2>
void transform_call22(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 input1, Iterator2 input2, Iterator2 input_map) {
    std::transform
    (
        policy,
        first1,
        last1,
        oneapi::dpl::make_transform_iterator
        (
            oneapi::dpl::make_permutation_iterator(input1, input_map),
            square()
        ),
        oneapi::dpl::make_permutation_iterator(input2, input_map),
        std::plus<int>()
    );
}

// transform(transform_it<zip_it<buffer x3>, func>, transform_it<zip_it<buffer x3>, func>, counting_it, buffer)
template <typename Policy, typename Iterator1, typename Iterator2>
void transform_call23(Policy policy, Iterator1 input1, Iterator1 input2, Iterator1 input3, Iterator1 input4, Iterator2 second_input) {
    std::transform
    (
        policy,
        oneapi::dpl::make_transform_iterator
        (
            oneapi::dpl::make_zip_iterator(input1, input2, input3),
            negate_tuple_components()
        ),
        oneapi::dpl::make_transform_iterator
        (
            oneapi::dpl::make_zip_iterator(input1 + 4, input2 + 4, input3 + 4),
            negate_tuple_components()
        ),
        second_input,
        input4,
        add_tuple_components4()
    );
}

// transform(buffer, buffer, zip_it<device_pointer x2>, device_pointer)
template <typename Policy, typename Iterator1, typename Iterator2, typename IteratorResult>
void transform_call24(Policy policy, Iterator1 first1, Iterator1 last1, Iterator2 zip_input1, Iterator2 zip_input2, IteratorResult result) {
// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// transform_iterator as a component of the permutation_iterator
#if 0
    std::transform
    (
        policy, first1, last1,
        oneapi::dpl::make_zip_iterator(zip_input1, zip_input2),
        result,
        add_tuple_components2()
    );
#endif
}

int main() {

    // used to detect failures
    int failed_tests = 0;
    int num_failing = 0;
    std::string test_name = "";

    // First Group: Testing std::transform with buffer as 1st parameter

    {
        // test 1/15

        // create buffers
        sycl::buffer<uint64_t, 1> src1_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src2_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src3_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src4_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src5_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src6_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> dst_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf { sycl::range<1>(4) };

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto src3_it = oneapi::dpl::begin(src3_buf);
        auto src4_it = oneapi::dpl::begin(src4_buf);
        auto src5_it = oneapi::dpl::begin(src5_buf);
        auto src6_it = oneapi::dpl::begin(src6_buf);
        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        iota_buffer(src1_buf, 0, 8, 0);
        iota_buffer(src2_buf, 0, 8, 10);
        iota_reverse_buffer(src3_buf, 0, 8, 0);
        iota_reverse_buffer(src4_buf, 0, 8, 10);
        iota_buffer(src5_buf, 0, 8, -10);
        iota_reverse_buffer(src6_buf, 0, 8, -10);
        fill_buffer(dst_buf, 0, 8, 0);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 3; map[1] = 2; map[2] = 0; map[3] = 1;
        }

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int hostArray2[8];
        int hostArray3[8];
        int hostArray4[8];
        int hostArray5[8];
        int hostArray6[8];
        int dstHostArray[8];
        int mapHostArray[4];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray2 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray3 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray4 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray5 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray6 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *dstDeviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *mapDeviceArray = (int*) malloc_device(4 * sizeof(int), dev, ctxt);

        // fill host arrays
        iota_array(hostArray, 0, 8, 0);
        iota_array(hostArray2, 0, 8, 10);
        iota_reverse_array(hostArray3, 0, 8, 0);
        iota_reverse_array(hostArray4, 0, 8, 10);
        iota_array(hostArray5, 0, 8, -10);
        iota_reverse_array(hostArray6, 0, 8, -10);
        fill_array(dstHostArray, 0, 8, 0);

        mapHostArray[0] = 3; mapHostArray[1] = 2; mapHostArray[2] = 0; mapHostArray[3] = 1;

        // copy host arrays to device arrays
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, hostArray2, deviceArray2);
        host_to_device(myQueue, hostArray3, deviceArray3);
        host_to_device(myQueue, hostArray4, deviceArray4);
        host_to_device(myQueue, hostArray5, deviceArray5);
        host_to_device(myQueue, hostArray6, deviceArray6);
        host_to_device(myQueue, dstHostArray, dstDeviceArray);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_dst = dpct::device_pointer<int>(dstDeviceArray);

            // transform(buffer, buffer, buffer, buffer)
            transform_call(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, src2_it, dst_it);
            transform_call(oneapi::dpl::execution::dpcpp_default, dptr, dptr + 4, dptr2, dptr_dst);
        }

        // copy device array back to host array
        device_to_host(myQueue, dstHostArray, dstDeviceArray);

        {
            test_name = "transform with buffer 1/15";
            auto dst = dst_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, dstHostArray[i], 10 + i*2);
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 10 + i*2);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, dstHostArray[i], 0);
                    num_failing += ASSERT_EQUAL(test_name, dst[i], 0);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/15

        iota_buffer(src1_buf, 0, 8, 0);
        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);

            // transform(buffer, buffer, buffer)
            transform_call(oneapi::dpl::execution::dpcpp_default, src1_it + 4, src1_it + 8, src1_it);
            transform_call(oneapi::dpl::execution::dpcpp_default, dptr + 4, dptr + 8, dptr);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with buffer 2/15";
            auto src = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], -4-i);
                    num_failing += ASSERT_EQUAL(test_name, src[i], -4 - i);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 3/15

        iota_buffer(src1_buf, 0, 8, 0);
        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);

            // transform(buffer, buffer, counting_iterator, buffer)
            transform_call(oneapi::dpl::execution::dpcpp_default, src1_it + 2, src1_it + 6, dpct::make_counting_iterator(3), src1_it + 4);
            transform_call(oneapi::dpl::execution::dpcpp_default, dptr + 2, dptr + 6, dpct::make_counting_iterator(3), dptr + 4);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with buffer 3/15";
            auto src = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], -3 + i*2);
                    num_failing += ASSERT_EQUAL(test_name, src[i], -3 + i*2);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 4/15

        iota_buffer(src1_buf, 0, 8, 0);
        iota_reverse_buffer(src2_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_reverse_array(hostArray2, 0, 8, 0);
        host_to_device(myQueue, hostArray2, deviceArray2);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(buffer, buffer, perm_it, buffer)
            transform_call2(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, src2_it, map_it, src1_it);
            transform_call2(oneapi::dpl::execution::dpcpp_default, dptr, dptr + 4, dptr2, dptr_map, dptr);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with buffer 4/15";
            auto src = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i == 0) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 4);
                    num_failing += ASSERT_EQUAL(test_name, src[i], 4);
                }
                else if (i == 1) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 6);
                    num_failing += ASSERT_EQUAL(test_name, src[i], 6);
                }
                else if (i == 2 || i == 3) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 9);
                    num_failing += ASSERT_EQUAL(test_name, src[i], 9);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 5/15

        iota_buffer(src1_buf, 0, 8, 0);
        iota_buffer(src2_buf, 0, 8, 10);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_array(hostArray2, 0, 8, 10);
        host_to_device(myQueue, hostArray2, deviceArray2);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr4 = dpct::device_pointer<int>(deviceArray4);

            // transform(buffer, buffer, zip_it<buffer x3>, buffer)
            transform_call3(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, src2_it, src3_it, src4_it, src1_it);
            transform_call3(oneapi::dpl::execution::dpcpp_default, dptr, dptr + 4, dptr2, dptr3, dptr4, dptr);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with buffer 5/15";
            auto src = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 34);
                    num_failing += ASSERT_EQUAL(test_name, src[i], 34);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 6/15

        iota_buffer(src1_buf, 0, 8, 0);
        iota_reverse_buffer(src2_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_reverse_array(hostArray2, 0, 8, 0);
        host_to_device(myQueue, hostArray2, deviceArray2);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);

            // transform(buffer, buffer, transform_it<zip_it<buffer x2>, func>, buffer)
            transform_call4(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, src2_it, square(), src1_it);
            transform_call4(oneapi::dpl::execution::dpcpp_default, dptr, dptr + 4, dptr2, square(), dptr);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with buffer 6/15";
            auto src = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], (7-i)*(7-i) + i);
                    num_failing += ASSERT_EQUAL(test_name, src[i], (7-i)*(7-i) + i);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 7/15

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 2; map[1] = 6; map[2] = 7; map[3] = 4;
        }

        mapHostArray[0] = 2; mapHostArray[1] = 6; mapHostArray[2] = 7; mapHostArray[3] = 4;
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(buffer, buffer, perm_it, buffer)
            transform_call5(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 3, src1_it + 1, src2_it, map_it);
            transform_call5(oneapi::dpl::execution::dpcpp_default, dptr, dptr + 3, dptr + 1, dptr2, dptr_map);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray2, deviceArray2);

        {
            test_name = "transform with buffer 7/15";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i == 2) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 1);
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 1);
                }
                else if (i == 6) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 3);
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 3);
                }
                else if (i == 7) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 5);
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 5);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 7-i);
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 7-i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 8/15
        // commented out due to zip_iterator as result sequence

        iota_buffer(src1_buf, 0, 8, 0); // all 0
        iota_buffer(src2_buf, 0, 8, 10);// 0 10 20 30...

        iota_array(hostArray, 0, 8, 0); //all 0
        host_to_device(myQueue, hostArray, deviceArray);
        iota_array(hostArray2, 0, 8, 10);// 0 10 20 30
        host_to_device(myQueue, hostArray2, deviceArray2);

        mapHostArray[0] = 5; mapHostArray[1] = 6; mapHostArray[2] = 4; mapHostArray[3] = 7;
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 5; map[1] = 6; map[2] = 4; map[3] = 7;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

// Comment out test using permutation_iterator until oneDPL is updated to better support use of
// permutation_iterator
#if 0
            // transform(buffer, buffer, buffer, zip_it<perm_it x2>)
            transform_call6(oneapi::dpl::execution::dpcpp_default, src1_it + 4, src1_it + 7, src1_it + 5, src2_it, src3_it, map_it);
            transform_call6(oneapi::dpl::execution::dpcpp_default, dptr + 4, dptr + 7, dptr + 5, dptr2, dptr3, dptr_map);
            if (i > 0 && i < 5) {
                num_failing += ASSERT_EQUAL(test_name, src2[i], i+16);
                num_failing += ASSERT_EQUAL(test_name, hostArray2[i], i+16);
            }

            else {
                num_failing += ASSERT_EQUAL(test_name, src2[i], i+10);
                num_failing += ASSERT_EQUAL(test_name, hostArray2[i], i+10);
            }
#endif
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray2, deviceArray2);
        device_to_host(myQueue, hostArray3, deviceArray3);

        {
            test_name = "transform with buffer 8/15";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            auto src3 = src3_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                std::cout << "src2[" << i << "] = " << src2[i] << " ";
                std::cout << "hostArray2[" << i << "] = " << hostArray2[i] << " ";
                std::cout << "src3[" << i << "] = " << src3[i] << std::endl;
                std::cout << "hostArray3[" << i << "] = " << hostArray3[i] << " ";
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 9/15

        iota_buffer(src1_buf, 0, 8, 0);
        iota_buffer(src2_buf, 0, 8, 10);
        iota_reverse_buffer(src3_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_array(hostArray2, 0, 8, 10);
        host_to_device(myQueue, hostArray2, deviceArray2);
        iota_reverse_array(hostArray3, 0, 8, 0);
        host_to_device(myQueue, hostArray3, deviceArray3);

        mapHostArray[0] = 1; mapHostArray[1] = 2; mapHostArray[2] = 3; mapHostArray[3] = 4;
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 1; map[1] = 2; map[2] = 3; map[3] = 4;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(buffer, buffer, zip_it<perm_it x2>, perm_it)
            transform_call7(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, src2_it, src3_it, map_it);
            transform_call7(oneapi::dpl::execution::dpcpp_default, dptr, dptr + 4, dptr2, dptr3, dptr_map);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray2, deviceArray2);
        device_to_host(myQueue, hostArray3, deviceArray3);

        {
            test_name = "transform with buffer 9/15";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i > 0 && i < 5) {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], i+16);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], i+16);
                }

                else {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], i+10);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], i+10);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 10/15

        iota_buffer(src1_buf, 0, 8, 0);
        iota_buffer(src2_buf, 0, 8, 10);
        iota_reverse_buffer(src3_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_array(hostArray2, 0, 8, 10);
        host_to_device(myQueue, hostArray2, deviceArray2);
        iota_reverse_array(hostArray3, 0, 8, 0);
        host_to_device(myQueue, hostArray3, deviceArray3);

        mapHostArray[0] = 4; mapHostArray[1] = 2; mapHostArray[2] = 5; mapHostArray[3] = 3;
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 4; map[1] = 2; map[2] = 5; map[3] = 3;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr4 = dpct::device_pointer<int>(deviceArray4);
            auto dptr5 = dpct::device_pointer<int>(deviceArray5);
            auto dptr6 = dpct::device_pointer<int>(deviceArray6);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(buffer, buffer, zip_it<perm_it, buffer x4>, buffer)
            transform_call8(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, src2_it, src3_it, src4_it, src5_it, src6_it, map_it);
            transform_call8(oneapi::dpl::execution::dpcpp_default, dptr, dptr + 4, dptr2, dptr3, dptr4, dptr5, dptr6, dptr_map);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray3, deviceArray3);

        {
            test_name = "transform with buffer 10/15";
            auto src3 = src3_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i == 0) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 25);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], 25);
                }
                else if (i == 1) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 22);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], 22);
                }
                else if (i == 2) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 24);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], 24);
                }
                else if (i == 3) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 21);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], 21);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 7-i);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], 7-i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 11/15
        // commented out due to zip_iterator as result sequence

        iota_reverse_buffer(src3_buf, 0, 8, 0);

        iota_reverse_array(hostArray3, 0, 8, 0);
        host_to_device(myQueue, hostArray3, deviceArray3);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);

            // transform(buffer, buffer, zip_it<buffer x2>, zip_it<buffer x2>)
            transform_call9(oneapi::dpl::execution::dpcpp_default, src1_it + 4, src1_it + 8, src2_it, src3_it);
            transform_call9(oneapi::dpl::execution::dpcpp_default, dptr + 4, dptr + 8, dptr2, dptr3);
        }

        device_to_host(myQueue, hostArray2, deviceArray2);
        device_to_host(myQueue, hostArray3, deviceArray3);

        {
            test_name = "transform with buffer 11/15";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            auto src3 = src3_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 14 + i*2);
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 11);
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 14 + i*2);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], 11);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 10 + i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 7-i);
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 10 + i);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], 7-i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 12/15

        iota_buffer(src1_buf, 0, 8, 0);

        fill_array(hostArray, 0, 8, 5);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);

            // transform(buffer, buffer, device_pointer<T>, buffer)
            transform_call(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, dptr, src1_it);
        }

        {
            test_name = "transform with buffer 12/15";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i + 5);
                else
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 13/15
        // runtime fail due to constant_iterator not changing output

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);

            // transform(buffer, buffer, constant_it, buffer)
            transform_call(oneapi::dpl::execution::dpcpp_default, src1_it + 2, src1_it + 6, dpct::constant_iterator<const int>(7), src1_it);
            transform_call(oneapi::dpl::execution::dpcpp_default, dptr + 2, dptr + 6, dpct::constant_iterator<const int>(7), dptr);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with buffer 13/15";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i > 1 && i < 6) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i+7);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i+7);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 14/15

        iota_buffer(src1_buf, 0, 8, 0);

        fill_array(hostArray, 0, 8, 5);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_reverse_array(hostArray2, 0, 8, 0);
        host_to_device(myQueue, hostArray2, deviceArray2);

        {
            auto dptr1 = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);

            // transform(buffer, buffer, device_pointer<T>, device_pointer<T>)
            transform_call(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, dptr1, dptr2);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray2, deviceArray2);

        test_name = "transform with buffer 14/15";
        for (int i = 0; i != 8; ++i) {
            if (i < 4)
                num_failing += ASSERT_EQUAL(test_name, hostArray2[i], i+5);
            else
                num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 7-i);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;

        // test 15/15

        fill_array(hostArray, 0, 8, 5);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_array(hostArray2, 0, 8, 0);
        host_to_device(myQueue, hostArray2, deviceArray2);
        iota_reverse_array(hostArray3, 0, 8, 0);
        host_to_device(myQueue, hostArray3, deviceArray3);

        {
            auto dptr1 = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);

            // transform(buffer, buffer, zip_it<device_pointer<T> x2>, device_pointer<T>)
            transform_call24(oneapi::dpl::execution::dpcpp_default, src1_it, src1_it + 4, dptr1, dptr2, dptr3);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray3, deviceArray3);

        test_name = "transform with buffer 15/15";
        for (int i = 0; i != 8; ++i) {
            if (i < 4)
                num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 5 + i*2);
            else
                num_failing += ASSERT_EQUAL(test_name, hostArray3[i], 7-i);
        }

        failed_tests += test_passed(num_failing, test_name);
        num_failing = 0;
    }


    /*** END OF FIRST GROUP **/


    // Second Group: Testing std::transform with make_permutation_iterator as 1st parameter

    {
        // test 1/8

        // create buffers

        sycl::buffer<uint64_t, 1> src1_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src2_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src3_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map2_buf { sycl::range<1>(8) };

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto src3_it = oneapi::dpl::begin(src3_buf);
        auto map_it = oneapi::dpl::begin(map_buf);
        auto map2_it = oneapi::dpl::begin(map2_buf);

        iota_buffer(src1_buf, 0, 8, 0);
        iota_reverse_buffer(src2_buf, 0, 8, 0);
        iota_buffer(src3_buf, 0, 8, 10);

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int hostArray2[8];
        int hostArray3[8];
        int mapHostArray[8];
        int mapHostArray2[8];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray2 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray3 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *mapDeviceArray = (int*) malloc_device(4 * sizeof(int), dev, ctxt);
        int *mapDeviceArray2 = (int*) malloc_device(4 * sizeof(int), dev, ctxt);

        // fill host arrays
        iota_array(hostArray, 0, 8, 0);
        iota_reverse_array(hostArray2, 0, 8, 0);
        iota_array(hostArray3, 0, 8, 10);

        mapHostArray[0] = 7; mapHostArray[1] = 6; mapHostArray[2] = 5; mapHostArray[3] = 4; mapHostArray[4] = 2; mapHostArray[5] = 3; mapHostArray[6] = 0; mapHostArray[7] = 1;

        mapHostArray2[0] = 0; mapHostArray2[1] = 2; mapHostArray2[2] = 1; mapHostArray2[3] = 3; mapHostArray2[4] = 7; mapHostArray2[5] = 4; mapHostArray2[6] = 6; mapHostArray2[7] = 5;

        // copy host arrays to device arrays
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, hostArray2, deviceArray2);
        host_to_device(myQueue, hostArray3, deviceArray3);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);
        host_to_device(myQueue, mapHostArray2, mapDeviceArray2);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 7; map[1] = 6; map[2] = 5; map[3] = 4; map[4] = 2; map[5] = 3; map[6] = 0; map[7] = 1;

            auto map2 = map2_it.get_buffer().template get_access<sycl::access::mode::write>();
            map2[0] = 0; map2[1] = 2; map2[2] = 1; map2[3] = 3; map2[4] = 7; map2[5] = 4; map2[6] = 6; map2[7] = 5;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(perm_it, perm_it, buffer, perm_it)
            transform_call10(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, map_it);
            transform_call10(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr_map);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);
        device_to_host(myQueue, hostArray2, deviceArray2);

        {
            test_name = "transform with make_perm_it 1/8";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i*2);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i*2);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/8

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);
            auto dptr_map2 = dpct::device_pointer<int>(mapDeviceArray2);

            // transform(perm_it<perm_it, buffer>, perm_it<perm_it, buffer>, perm_it, perm_it<perm_it, buffer>)
            transform_call11(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, map_it, map2_it);
            transform_call11(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr_map, dptr_map2);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with make_perm_it 2/8";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i == 0) {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], 5);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 5);
                }
                else if (i == 1) {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], 8);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 8);
                }
                else if (i == 2) {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], 6);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 6);
                }
                else if (i == 3) {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], 9);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 9);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 3/8

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);

        mapHostArray[0] = 4; mapHostArray[1] = 3; mapHostArray[2] = 5; mapHostArray[3] = 6;
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 4; map[1] = 3; map[2] = 5; map[3] = 6;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(perm_it, perm_it, transform_it<zip_it<buffer x2>, func>, perm_it)
            transform_call12(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, map_it);
            transform_call12(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr_map);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with make_perm_it 3/8";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                std::cout << src1[i] << " ";
                std::cout << hostArray[i] << std::endl;
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 4/8

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);

        mapHostArray[0] = 4; mapHostArray[1] = 3; mapHostArray[2] = 1; mapHostArray[3] = 2;
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 4; map[1] = 3; map[2] = 1; map[3] = 2;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(perm_it, perm_it, zip_it<perm_it x2>, perm_it)
            transform_call13(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, src3_it, map_it);
            transform_call13(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr3, dptr_map);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with make_perm_it 4/8";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i > 0 && i < 5) {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i+17);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i+17);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 5/8

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);

        mapHostArray[0] = 4; mapHostArray[1] = 3; mapHostArray[2] = 5; mapHostArray[3] = 6;
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 4; map[1] = 3; map[2] = 5; map[3] = 6;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(perm_it, perm_it, transform_it<perm_it, func>, perm_it)
            transform_call14(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, map_it);
            transform_call14(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr_map);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with make_perm_it 5/8";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i > 2 && i < 7) {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i + (7-i)*(7-i));
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i + (7-i)*(7-i));
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 6/8

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);

        mapHostArray[0] = 1; mapHostArray[1] = 2; mapHostArray[2] = 3; mapHostArray[3] = 0;
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 1; map[1] = 2; map[2] = 3; map[3] = 0;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(perm_it, perm_it, perm_it)
            transform_call15(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, map_it);
            transform_call15(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr_map);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray2, deviceArray2);

        {
            test_name = "transform with make_perm_it 6/8";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], -i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], -i);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 7-i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 7-i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 7/8

        iota_buffer(src1_buf, 0, 8, 0);
        iota_reverse_buffer(src2_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        iota_reverse_array(hostArray2, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, hostArray2, deviceArray2);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 1; map[1] = 2; map[2] = 3; map[3] = 0;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(perm_it, perm_it, buffer)
            transform_call16(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, map_it);
            transform_call16(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr_map);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray2, deviceArray2);

        {
            test_name = "transform with make_perm_it 7/8";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 3) {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], -i-1);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], -i-1);
                }
                else if (i == 3){
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 0);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 0);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 7-i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 7-i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 8/8

        iota_buffer(src1_buf, 0, 8, 0);
        iota_reverse_buffer(src2_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);
        iota_reverse_array(hostArray2, 0, 8, 0);
        host_to_device(myQueue, hostArray2, deviceArray2);

        mapHostArray2[0] = 6; mapHostArray2[1] = 7; mapHostArray2[2] = 5; mapHostArray2[3] = 4;
        host_to_device(myQueue, mapHostArray2, mapDeviceArray2);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 1; map[1] = 2; map[2] = 3; map[3] = 0;

            auto map2 = map2_it.get_buffer().template get_access<sycl::access::mode::write>();
            map2[0] = 6; map2[1] = 7; map2[2] = 5; map2[3] = 4;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);
            auto dptr_map2 = dpct::device_pointer<int>(mapDeviceArray2);

            // transform(perm_it, perm_it, perm_it, buffer)
            transform_call17(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, map_it, map2_it);
            transform_call17(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr_map, dptr_map2);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray2, deviceArray2);

        {
            test_name = "transform with make_perm_it 8/8";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i == 0) {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 7);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 7);
                }
                else if (i == 1) {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 9);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 9);
                }
                else if (i == 2) {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 8);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 8);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 7-i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 7-i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }


    /*** END OF SECOND GROUP ***/


    // Third Group: Testing std::transform with make_counting_iterator as 1st parameter

    {
        // test 1/4

        // create buffers
        sycl::buffer<uint64_t, 1> src1_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src2_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src3_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf { sycl::range<1>(4) };

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto src3_it = oneapi::dpl::begin(src3_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        iota_buffer(src1_buf, 0, 8, 0);
        iota_reverse_buffer(src2_buf, 0, 8, 0);
        iota_buffer(src3_buf, 0, 8, 10);

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int hostArray2[8];
        int hostArray3[8];
        int mapHostArray[4];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray2 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray3 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *mapDeviceArray = (int*) malloc_device(4 * sizeof(int), dev, ctxt);

        // fill host arrays
        iota_array(hostArray, 0, 8, 0);
        iota_reverse_array(hostArray2, 0, 8, 0);
        iota_array(hostArray3, 0, 8, 10);

        mapHostArray[0] = 3; mapHostArray[1] = 5; mapHostArray[2] = 7; mapHostArray[3] = 1;

        // copy host arrays to device arrays
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, hostArray2, deviceArray2);
        host_to_device(myQueue, hostArray3, deviceArray3);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 3; map[1] = 5; map[2] = 7; map[3] = 1;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(counting_it, counting_it, perm_it, perm_it)
            transform_call18(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(0), dpct::make_counting_iterator(0) + 4, src1_it, map_it);
            transform_call18(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(0), dpct::make_counting_iterator(0) + 4, dptr, dptr_map);
        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with make_counting_it 1/4";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i == 1) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 4);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], 4);
                }
                else if (i == 5) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 6);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], 6);
                }
                else if (i == 7) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 9);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], 9);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/4

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);

            // transform(counting_it, counting_it, buffer)
            transform_call(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(1), dpct::make_counting_iterator(1) + 4, src1_it);
            transform_call(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(1), dpct::make_counting_iterator(1) + 4, dptr);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with make_counting_it 2/4";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], -i-1);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], -i-1);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], i);
                    num_failing += ASSERT_EQUAL(test_name, src1[i], i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 3/4

        iota_buffer(src1_buf, 0, 8, 0);

        iota_array(hostArray, 0, 8, 0);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);

            // transform(counting_it, counting_it, zip_it<buffer x2>, buffer)
            transform_call19(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(0), dpct::make_counting_iterator(0) + 4, src1_it, src2_it, src3_it);
            transform_call19(oneapi::dpl::execution::dpcpp_default, dpct::make_counting_iterator(0), dpct::make_counting_iterator(0) + 4, dptr, dptr2, dptr3);

        }

        // copy device array back to host array
        device_to_host(myQueue, hostArray3, deviceArray3);

        {
            test_name = "transform with make_counting_it 3/4";
            auto src3 = src3_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], i+7);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], i+7);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, hostArray3[i], i+10);
                    num_failing += ASSERT_EQUAL(test_name, src3[i], i+10);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 4/4

        // fill hostArray with 5s
        fill_array(hostArray, 0, 8, 5);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);

            // transform(counting_it, counting_it, device_pointer<T>)
            transform_call
            (
                oneapi::dpl::execution::dpcpp_default,
                dpct::make_counting_iterator(0),
                dpct::make_counting_iterator(0) + 4,
                dptr + 2
            );
        }

        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with make_counting_it 4/4";
            for (int i = 0; i != 8; ++i) {
                if (i > 1 && i < 6)
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], -i+2);
                else
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 5);
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }


    /*** END OF THIRD GROUP ***/


    // Fourth Group: Testing std::transform with make_zip_iterator as 1st parameter

    {

        // test 1/2

        // runtime hang with zip_it<buffer x2, perm_it x2>

        // create buffers
        sycl::buffer<uint64_t, 1> src1_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src2_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src3_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src4_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src5_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src6_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf { sycl::range<1>(4) };
        sycl::buffer<uint64_t, 1> map2_buf { sycl::range<1>(4) };

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto src3_it = oneapi::dpl::begin(src3_buf);
        auto src4_it = oneapi::dpl::begin(src4_buf);
        auto src5_it = oneapi::dpl::begin(src5_buf);
        auto src6_it = oneapi::dpl::begin(src6_buf);
        auto map_it = oneapi::dpl::begin(map_buf);
        auto map2_it = oneapi::dpl::begin(map_buf);

        iota_buffer(src1_buf, 0, 8, 0);
        iota_reverse_buffer(src2_buf, 0, 8, 0);
        iota_buffer(src3_buf, 0, 8, 10);
        iota_reverse_buffer(src4_buf, 0, 8, 10);
        iota_buffer(src5_buf, 0, 8, -10);
        iota_reverse_buffer(src6_buf, 0, 8, -10);

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int hostArray2[8];
        int hostArray3[8];
        int hostArray4[8];
        int hostArray5[8];
        int hostArray6[8];
        int mapHostArray[8];
        int mapHostArray2[4];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray2 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray3 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray4 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray5 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray6 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *mapDeviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *mapDeviceArray2 = (int*) malloc_device(4 * sizeof(int), dev, ctxt);

        // fill host arrays
        iota_array(hostArray, 0, 8, 0);
        iota_array(hostArray2, 0, 8, 10);
        iota_reverse_array(hostArray3, 0, 8, 0);
        iota_reverse_array(hostArray4, 0, 8, 10);
        iota_array(hostArray5, 0, 8, -10);
        iota_reverse_array(hostArray6, 0, 8, -10);

        mapHostArray[0] = 5; mapHostArray[1] = 6; mapHostArray[2] = 7; mapHostArray[3] = 4;
        mapHostArray2[0] = 0; mapHostArray2[1] = 2; mapHostArray2[2] = 1; mapHostArray2[3] = 3;

        // copy host arrays to device arrays
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, hostArray2, deviceArray2);
        host_to_device(myQueue, hostArray3, deviceArray3);
        host_to_device(myQueue, hostArray4, deviceArray4);
        host_to_device(myQueue, hostArray5, deviceArray5);
        host_to_device(myQueue, hostArray6, deviceArray6);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);
        host_to_device(myQueue, mapHostArray2, mapDeviceArray2);
        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 5; map[1] = 6; map[2] = 7; map[3] = 4;

            auto map2 = map2_it.get_buffer().template get_access<sycl::access::mode::write>();
            map2[0] = 0; map2[1] = 2; map2[2] = 1; map2[3] = 3;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr4 = dpct::device_pointer<int>(deviceArray4);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);
            auto dptr_map2 = dpct::device_pointer<int>(mapDeviceArray2);

            // transform(zip_it<buffer x2, perm_it x2>, zip_it<buffer x2, perm_it x2>, buffer)
            transform_call20(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, src3_it, src4_it, map_it, map2_it);
            transform_call20(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr3, dptr4, dptr_map, dptr_map2);

        }

        // copy device back to host
        device_to_host(myQueue, hostArray4, deviceArray4);

        {
            test_name = "transform with make_zip_it 1/2";
            auto src4 = src4_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                std::cout << src4[i] << " ";
                std::cout << hostArray4[i] << std::endl;
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }

        // test 2/2

        // commented out due to zip_iterator as result sequence

        mapHostArray[0] = 4; mapHostArray[1] = 2; mapHostArray[2] = 5; mapHostArray[3] = 3;

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 4; map[1] = 2; map[2] = 5; map[3] = 3;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr4 = dpct::device_pointer<int>(deviceArray4);
            auto dptr5 = dpct::device_pointer<int>(deviceArray5);
            auto dptr6 = dpct::device_pointer<int>(deviceArray6);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(zip_it<perm_it x2>, zip_it<perm_it x2>, zip_it<perm_it x4>, zip_it<perm_it x2>)
            transform_call21(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, src3_it, src4_it, src5_it, src6_it, map_it);
            transform_call21(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr3, dptr4, dptr5, dptr6, dptr_map);
        }

        // copy device back to host
        device_to_host(myQueue, hostArray, deviceArray);
        device_to_host(myQueue, hostArray2, deviceArray2);

        {
            test_name = "transform with make_zip_it 2/2";
            auto src1 = src1_it.get_buffer().template get_access<sycl::access::mode::read>();
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                std::cout << src1[i] << " ";
                std::cout << src2[i] << " ";
                std::cout << hostArray[i] << " ";
                std::cout << hostArray2[i] << std::endl;
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }


    /*** END OF FOURTH GROUP ***/


    // Fifth Group: Testing std::transform with make_constant_iterator as 1st parameter

    {
        // test 1/1

        // runtime fail due to make_constant_iterator bug

        // create buffers
        sycl::buffer<uint64_t, 1> src1_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src2_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> map_buf { sycl::range<1>(4) };

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto map_it = oneapi::dpl::begin(map_buf);

        iota_buffer(src1_buf, 0, 8, 0);
        iota_buffer(src2_buf, 0, 8, 10);

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int hostArray2[8];
        int mapHostArray[4];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray2 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *mapDeviceArray = (int*) malloc_device(4 * sizeof(int), dev, ctxt);

        // fill host arrays
        iota_array(hostArray, 0, 8, 0);
        iota_array(hostArray2, 0, 8, 10);

        mapHostArray[0] = 4; mapHostArray[1] = 2; mapHostArray[2] = 3; mapHostArray[3] = 1;

        // copy host arrays to device arrays
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, hostArray2, deviceArray2);
        host_to_device(myQueue, mapHostArray, mapDeviceArray);

        {
            auto map = map_it.get_buffer().template get_access<sycl::access::mode::write>();
            map[0] = 4; map[1] = 2; map[2] = 3; map[3] = 1;
        }

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr_map = dpct::device_pointer<int>(mapDeviceArray);

            // transform(constant_it, constant_it, transform_it<perm_it, func>, perm_it)
            transform_call22(oneapi::dpl::execution::dpcpp_default, dpct::make_constant_iterator<const int>(5), dpct::make_constant_iterator<const int>(5) + 4, src1_it, src2_it, map_it);
            transform_call22(oneapi::dpl::execution::dpcpp_default, dpct::make_constant_iterator<const int>(5), dpct::make_constant_iterator<const int>(5) + 4, dptr, dptr2, dptr_map);

        }

        // copy device back to host
        device_to_host(myQueue, hostArray2, deviceArray2);

        {
            test_name = "transform with make_constant_it";
            auto src2 = src2_it.get_buffer().template get_access<sycl::access::mode::read>();

            for (int i = 0; i != 8; ++i) {
                if (i > 0 && i < 5) {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], 30);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], 30);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src2[i], i+10);
                    num_failing += ASSERT_EQUAL(test_name, hostArray2[i], i+10);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }


    /*** END OF FIFTH GROUP ***/


    // Sixth Group: Testing std::transform with make_transform_iterator as 1st parameter

    {
        // test 1/1

        // create buffers
        sycl::buffer<uint64_t, 1> src1_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src2_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src3_buf { sycl::range<1>(8) };
        sycl::buffer<uint64_t, 1> src4_buf { sycl::range<1>(8) };

        auto src1_it = oneapi::dpl::begin(src1_buf);
        auto src2_it = oneapi::dpl::begin(src2_buf);
        auto src3_it = oneapi::dpl::begin(src3_buf);
        auto src4_it = oneapi::dpl::begin(src4_buf);

        iota_buffer(src1_buf, 0, 8, 0);
        iota_buffer(src2_buf, 0, 8, 10);
        iota_reverse_buffer(src3_buf, 0, 8, 0);
        iota_reverse_buffer(src4_buf, 0, 8, 10);

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int hostArray2[8];
        int hostArray3[8];
        int hostArray4[8];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray2 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray3 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);
        int *deviceArray4 = (int*) malloc_device(8 * sizeof(int), dev, ctxt);

        // fill host arrays
        iota_array(hostArray, 0, 8, 0);
        iota_array(hostArray2, 0, 8, 10);
        iota_reverse_array(hostArray3, 0, 8, 0);
        iota_reverse_array(hostArray4, 0, 8, 10);

        // copy host arrays to device arrays
        host_to_device(myQueue, hostArray, deviceArray);
        host_to_device(myQueue, hostArray2, deviceArray2);
        host_to_device(myQueue, hostArray3, deviceArray3);
        host_to_device(myQueue, hostArray4, deviceArray4);

        {
            auto dptr = dpct::device_pointer<int>(deviceArray);
            auto dptr2 = dpct::device_pointer<int>(deviceArray2);
            auto dptr3 = dpct::device_pointer<int>(deviceArray3);
            auto dptr4 = dpct::device_pointer<int>(deviceArray4);

            // transform(transform_it<zip_it<buffer x3>, func>, transform_it<zip_it<buffer x3>, func>, counting_it, buffer)
            transform_call23(oneapi::dpl::execution::dpcpp_default, src1_it, src2_it, src3_it, src4_it, dpct::make_counting_iterator(0));
            transform_call23(oneapi::dpl::execution::dpcpp_default, dptr, dptr2, dptr3, dptr4, dpct::make_counting_iterator(0));

        }

        // copy device back to host
        device_to_host(myQueue, hostArray4, deviceArray4);

        {
            test_name = "transform with make_transform_it";
            auto src4 = src4_it.get_buffer().template get_access<sycl::access::mode::read>();
            for (int i = 0; i != 8; ++i) {
                if (i < 4) {
                    num_failing += ASSERT_EQUAL(test_name, src4[i], 0);
                    num_failing += ASSERT_EQUAL(test_name, hostArray4[i], 0);
                }
                else {
                    num_failing += ASSERT_EQUAL(test_name, src4[i], 17-i);
                    num_failing += ASSERT_EQUAL(test_name, hostArray4[i], 17-i);
                }
            }

            failed_tests += test_passed(num_failing, test_name);
            num_failing = 0;
        }
    }


    /*** END OF SIXTH GROUP ***/


    // Seventh Group: Testing std::transform with device_pointer<T> as 1st parameter

    {
        // test 1/1

        // runtime fail due to constant_iterator bug

        // create queue
        sycl::queue myQueue;
        auto dev = myQueue.get_device();
        auto ctxt = myQueue.get_context();

        // create host and device arrays
        int hostArray[8];
        int *deviceArray = (int*) malloc_device(8 * sizeof(int), dev, ctxt);

        // fill hostArray
        fill_array(hostArray, 0, 8, 5);
        host_to_device(myQueue, hostArray, deviceArray);

        {
            auto dptr_begin = dpct::device_pointer<int>(deviceArray);
            auto dptr_end = dpct::device_pointer<int>(deviceArray + 4);

            // transform(device_pointer<T>, device_pointer<T>, constant_it, device_pointer<T>)
            transform_call
            (
                oneapi::dpl::execution::dpcpp_default,
                dptr_begin,
                dptr_end,
                dpct::constant_iterator<const int>(6),
                dptr_begin
            );
        }

        device_to_host(myQueue, hostArray, deviceArray);

        {
            test_name = "transform with device_pointer<T>";
            for (int i = 0; i != 8; ++i) {
                if (i < 4)
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 11);
                else
                    num_failing += ASSERT_EQUAL(test_name, hostArray[i], 5);
            }

            failed_tests += test_passed(num_failing, test_name);
        }
    }


    /*** END OF SEVENTH GROUP ***/


    std::cout << std::endl << failed_tests << " failing test(s) detected." << std::endl;
    if (failed_tests == 0) {
        return 0;
    }
    return 1;
}
