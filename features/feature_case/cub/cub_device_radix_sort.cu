#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

template <typename T> inline T *safe_device_malloc(size_t Num = 1) {
  T *Ptr = nullptr;
  cudaError_t Err = cudaMalloc<T>(&Ptr, sizeof(T) * Num);
  if (Err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(Err) << std::endl;
    abort();
  }
  return Ptr;
}

inline void safe_host_copy_to_device(void *Dst, const void *Src, size_t Size) {
  cudaError_t Err = cudaMemcpy(Dst, Src, Size, cudaMemcpyHostToDevice);
  if (Err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(Err) << std::endl;
    abort();
  }
}

inline void safe_device_copy_to_host(void *Dst, const void *Src, size_t Size) {
  cudaError_t Err = cudaMemcpy(Dst, Src, Size, cudaMemcpyDeviceToHost);
  if (Err != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(Err) << std::endl;
    abort();
  }
}

inline void safe_sync(cudaStream_t stream) {
  cudaError_t Err = cudaStreamSynchronize(stream);
  if (Err != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(Err)
              << std::endl;
    abort();
  }
}

inline cudaStream_t safe_create_stream() {
  cudaStream_t stream;
  cudaError_t Err = cudaStreamCreate(&stream);
  if (Err != cudaSuccess) {
    std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(Err)
              << std::endl;
    abort();
  }
  return stream;
}

template <int N> struct uint_byte_map {};
template <> struct uint_byte_map<1> {
  using type = uint8_t;
};
template <> struct uint_byte_map<2> {
  using type = uint16_t;
};
template <> struct uint_byte_map<4> {
  using type = uint32_t;
};
template <> struct uint_byte_map<8> {
  using type = uint64_t;
};

template <typename T> struct uint_map {
  using type = typename uint_byte_map<sizeof(T)>::type;
};

template <typename T, typename OutKeyT> class translate_key {
  using uint_type_t = typename uint_map<T>::type;

public:
  translate_key(int begin_bit, int end_bit) {
    shift = begin_bit;
    mask = ~OutKeyT(0); // all ones
    mask = mask >> (sizeof(OutKeyT) * 8 -
                    (end_bit - begin_bit));           // setup appropriate mask
    flip_sign = 1UL << (sizeof(uint_type_t) * 8 - 1); // sign bit
    flip_key = ~uint_type_t(0);                       // 0xF...F
  }

  inline OutKeyT operator()(const T &key) const {
    uint_type_t intermediate;
    if constexpr (std::is_floating_point<T>::value) {
      // normal case (both -0.0f and 0.0f equal -0.0f)
      if (key != T(-0.0f)) {
        uint_type_t is_negative = reinterpret_cast<const uint_type_t &>(key) >>
                                  (sizeof(uint_type_t) * 8 - 1);
        intermediate = reinterpret_cast<const uint_type_t &>(key) ^
                       ((is_negative * flip_key) | flip_sign);
      } else // special case for -0.0f to keep stability with 0.0f
      {
        T negzero = T(-0.0f);
        intermediate = reinterpret_cast<const uint_type_t &>(negzero);
      }
    } else if constexpr (std::is_signed<T>::value) {
      intermediate = reinterpret_cast<const uint_type_t &>(key) ^ flip_sign;
    } else {
      intermediate = key;
    }

    return static_cast<OutKeyT>(intermediate >> shift) &
           mask; // shift, cast, and mask
  }

private:
  uint8_t shift;
  OutKeyT mask;
  uint_type_t flip_sign;
  uint_type_t flip_key;
};

template <typename T> struct key_loader {
  using type = T;
  const type &operator()(const T &v) const { return v; }
};
template <typename KeyTp, typename ValueTp>
struct key_loader<std::pair<KeyTp, ValueTp>> {
  using type = KeyTp;
  const type &operator()(const std::pair<KeyTp, ValueTp> &v) const {
    return v.first;
  }
};

template <typename Iter>
void transform_sort(Iter begin, Iter end, size_t n, bool descending,
                    int begin_bit, int end_bit) {
  using ValueTp = typename std::iterator_traits<Iter>::value_type;
  using SortKeyTp = typename key_loader<ValueTp>::type;

  key_loader<ValueTp> load;
  int clipped_begin_bit = ::std::max(begin_bit, 0);
  int clipped_end_bit =
      ::std::min((::std::uint64_t)end_bit, sizeof(SortKeyTp) * 8);
  int num_bytes = (clipped_end_bit - clipped_begin_bit - 1) / 8 + 1;

  auto transform_sort_fn = [&](auto x) {
    using TransTp = typename std::decay_t<decltype(x)>;
    auto trans_key = translate_key<SortKeyTp, TransTp>(begin_bit, end_bit);
    auto sort_with_compare = [&](const auto &comp) {
      return ::std::sort(begin, end, [=](auto a, auto b) {
        return comp(trans_key(load(a)), trans_key(load(b)));
      });
    };

    if (descending)
      sort_with_compare(::std::greater<TransTp>());
    else
      sort_with_compare(::std::less<TransTp>());
  };

  if (num_bytes == 1) {
    transform_sort_fn.template operator()<uint8_t>(0);
  } else if (num_bytes == 2) {
    transform_sort_fn.template operator()<uint16_t>(0);
  } else if (num_bytes <= 4) {
    transform_sort_fn.template operator()<uint32_t>(0);
  } else { // if (num_bytes <= 8)
    transform_sort_fn.template operator()<uint64_t>(0);
  }
}

template <typename T> struct compare_if_equal {
  bool operator()(const T &a, const T &b) const { return a == b; }
};

template <> struct compare_if_equal<float> {
  static constexpr double EPS = 1e-6;
  bool operator()(float a, float b) const { return std::abs(a - b) < EPS; }
};

template <> struct compare_if_equal<double> {
  static constexpr double EPS = 1e-6;
  bool operator()(float a, float b) const { return std::abs(a - b) < EPS; }
};

template <typename KeyTp, typename ValueTp>
bool check_sort_pairs(const KeyTp *keys_in, const ValueTp *values_in, size_t n,
                      bool descending = false, int begin_bit = 0,
                      int end_bit = sizeof(KeyTp) * 8) {
  std::vector<KeyTp> keys_out(n, KeyTp{});
  std::vector<ValueTp> values_out(n, ValueTp{});
  KeyTp *d_key_in = safe_device_malloc<KeyTp>(n);
  KeyTp *d_key_out = safe_device_malloc<KeyTp>(n);
  ValueTp *d_value_in = safe_device_malloc<ValueTp>(n);
  ValueTp *d_value_out = safe_device_malloc<ValueTp>(n);
  safe_host_copy_to_device(d_key_in, keys_in, sizeof(KeyTp) * n);
  safe_host_copy_to_device(d_value_in, values_in, sizeof(ValueTp) * n);
  cudaStream_t stream = safe_create_stream();
  void *temp_storage = nullptr;
  size_t temp_storage_size = 0;

  if (descending) {
    cub::DeviceRadixSort::SortPairsDescending(
        temp_storage, temp_storage_size, d_key_in, d_key_out, d_value_in,
        d_value_out, n, begin_bit, end_bit, stream);
    temp_storage = safe_device_malloc<char>(temp_storage_size);
    cub::DeviceRadixSort::SortPairsDescending(
        temp_storage, temp_storage_size, d_key_in, d_key_out, d_value_in,
        d_value_out, n, begin_bit, end_bit, stream);
  } else {
    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_key_in,
                                    d_key_out, d_value_in, d_value_out, n,
                                    begin_bit, end_bit, stream);
    temp_storage = safe_device_malloc<char>(temp_storage_size);
    cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_size, d_key_in,
                                    d_key_out, d_value_in, d_value_out, n,
                                    begin_bit, end_bit, stream);
  }
  safe_sync(stream);
  safe_device_copy_to_host(keys_out.data(), d_key_out, sizeof(KeyTp) * n);
  safe_device_copy_to_host(values_out.data(), d_value_out, sizeof(ValueTp) * n);
  cudaFree(temp_storage);
  cudaFree(d_key_in);
  cudaFree(d_value_in);
  cudaFree(d_key_out);
  cudaFree(d_value_out);

  std::vector<std::pair<KeyTp, ValueTp>> vec;
  for (size_t i = 0; i < n; ++i)
    vec.emplace_back(keys_in[i], values_in[i]);

  compare_if_equal<KeyTp> comp_key;
  compare_if_equal<ValueTp> comp_val;
  transform_sort(vec.begin(), vec.end(), n, descending, begin_bit, end_bit);

  for (size_t i = 0; i < n; ++i) {
    if (!comp_key(vec[i].first, keys_out[i]) || !comp_val(vec[i].second, values_out[i]))
      return false;
  }
  
  return true;
}

template <typename KeyTp>
bool check_sort_keys(const KeyTp *keys_in, size_t n, bool descending = false,
                     int begin_bit = 0, int end_bit = sizeof(KeyTp) * 8) {
  std::vector<KeyTp> keys_out(n, KeyTp{});
  KeyTp *d_key_in = safe_device_malloc<KeyTp>(n);
  KeyTp *d_key_out = safe_device_malloc<KeyTp>(n);
  safe_host_copy_to_device(d_key_in, keys_in, sizeof(KeyTp) * n);
  cudaStream_t stream = safe_create_stream();
  void *temp_storage = nullptr;
  size_t temp_storage_size = 0;

  if (descending) {
    cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size,
                                             d_key_in, d_key_out, n, begin_bit,
                                             end_bit, stream);
    temp_storage = safe_device_malloc<char>(temp_storage_size);
    cub::DeviceRadixSort::SortKeysDescending(temp_storage, temp_storage_size,
                                             d_key_in, d_key_out, n, begin_bit,
                                             end_bit, stream);
  } else {
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, d_key_in,
                                   d_key_out, n, begin_bit, end_bit, stream);
    temp_storage = safe_device_malloc<char>(temp_storage_size);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_size, d_key_in,
                                   d_key_out, n, begin_bit, end_bit, stream);
  }
  safe_sync(stream);
  safe_device_copy_to_host(keys_out.data(), d_key_out, sizeof(KeyTp) * n);
  cudaFree(temp_storage);
  cudaFree(d_key_in);
  cudaFree(d_key_out);

  std::vector<KeyTp> vec;
  for (size_t i = 0; i < n; ++i)
    vec.push_back(keys_in[i]);
  transform_sort(vec.begin(), vec.end(), n, descending, begin_bit, end_bit);
  return std::equal(keys_out.begin(), keys_out.end(), vec.begin(),
                    compare_if_equal<KeyTp>());
}

bool sort_keys_simple(void) {
  int n = 7;
  int in[] = {8, 6, 7, 5, 3, 0, 9};
  if (!check_sort_keys(in, n)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool sort_keys_descending_simple(void) {
  int n = 7;
  int in[] = {8, 6, 7, 5, 3, 0, 9};
  if (!check_sort_keys(in, n, true)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool sort_pairs_simple(void) {
  int n = 7;
  int keys[] = {8, 6, 7, 5, 3, 0, 9};
  int values[] = {0, 1, 2, 3, 4, 5, 6};
  if (!check_sort_pairs(keys, values, n)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool sort_pairs_descending_simple(void) {
  int n = 7;
  int keys[] = {8, 6, 7, 5, 3, 0, 9};
  int values[] = {0, 1, 2, 3, 4, 5, 6};
  if (!check_sort_pairs(keys, values, n, true)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool sort_keys_bit_range(void) {
  int n = 7;
  int in[] = {8, 6, 7, 5, 3, 0, 9};
  if (!check_sort_keys(in, n, false, 4, 8)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool sort_keys_decending_bit_range(void) {
  int n = 7;
  int in[] = {8, 6, 7, 5, 3, 0, 9};
  if (!check_sort_keys(in, n, true, 4, 8)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool sort_pairs_bit_range(void) {
  int n = 7;
  int keys[] = {8, 6, 7, 5, 3, 0, 9};
  int values[] = {0, 1, 2, 3, 4, 5, 6};
  if (!check_sort_pairs(keys, values, n, false, 4, 8)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool sort_pairs_decending_bit_range(void) {
  int n = 7;
  int keys[] = {8, 6, 7, 5, 3, 0, 9};
  int values[] = {0, 1, 2, 3, 4, 5, 6};
  if (!check_sort_pairs(keys, values, n, true, 4, 8)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool check_sort_keys_float(void) {
  int n = 7;
  float keys[] = {8.1, 6.5, 7.3, 5.5, 3.14, 0.7, 9.9};
  if (!check_sort_keys(keys, n, false)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool check_sort_keys_double(void) {
  int n = 7;
  double keys[] = {8.1, 6.5, 7.3, 5.5, 3.14, 0.7, 9.9};
  if (!check_sort_keys(keys, n, true, 4, 8)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool check_sort_pairs_float(void) {
  int n = 7;
  float keys[] = {8.1, 6.5, 7.3, 5.5, 3.14, 0.7, 9.9};
  int values[] = {0, 1, 2, 3, 4, 5, 6};
  if (!check_sort_pairs(keys, values, n, false)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

bool check_sort_pairs_double(void) {
  int n = 7;
  double keys[] = {8.1, 6.5, 7.3, 5.5, 3.14, 0.7, 9.9};
  int values[] = {0, 1, 2, 3, 4, 5, 6};
  if (!check_sort_pairs(keys, values, n, true, 4, 8)) {
    std::cerr << __func__ << " fun failed\n";
    return false;
  }
  return true;
}

typedef bool (*test_fn)(void);

static std::vector<test_fn> functions = {
    sort_keys_simple,       sort_keys_descending_simple,
    sort_pairs_simple,      sort_pairs_descending_simple,
    sort_keys_bit_range,    sort_keys_decending_bit_range,
    sort_pairs_bit_range,   sort_pairs_decending_bit_range,
    check_sort_keys_float,  check_sort_keys_double,
    check_sort_pairs_float, check_sort_pairs_double};

int main() {
  bool Result = true;
  for (const auto &fn : functions)
    Result = fn() && Result;
  if (!Result) {
    std::cerr << "test failed\n";
    return 1;
  }
  return 0;
}