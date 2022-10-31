#include "cusolverDn.h"

#include <cmath>
#include <vector>
#include <cstdio>
#include <complex>

template<class d_data_t>
struct Data {
  float *h_data;
  d_data_t *d_data;
  int element_num;
  Data(int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    memset(h_data, 0, sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
  }
  Data(float* input_data, int element_num) : element_num(element_num) {
    h_data = (float*)malloc(sizeof(float) * element_num);
    cudaMalloc(&d_data, sizeof(d_data_t) * element_num);
    cudaMemset(d_data, 0, sizeof(d_data_t) * element_num);
    memcpy(h_data, input_data, sizeof(float) * element_num);
  }
  ~Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    from_float_convert(h_data, h_temp);
    cudaMemcpy(d_data, h_temp, sizeof(d_data_t) * element_num, cudaMemcpyHostToDevice);
    free(h_temp);
  }
  void D2H() {
    d_data_t* h_temp = (d_data_t*)malloc(sizeof(d_data_t) * element_num);
    memset(h_temp, 0, sizeof(d_data_t) * element_num);
    cudaMemcpy(h_temp, d_data, sizeof(d_data_t) * element_num, cudaMemcpyDeviceToHost);
    to_float_convert(h_temp, h_data);
    free(h_temp);
  }
private:
  inline void from_float_convert(float* in, d_data_t* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
  inline void to_float_convert(d_data_t* in, float* out) {
    for (int i = 0; i < element_num; i++)
      out[i] = in[i];
  }
};
template <>
inline void Data<float2>::from_float_convert(float* in, float2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}
template <>
inline void Data<double2>::from_float_convert(float* in, double2* out) {
  for (int i = 0; i < element_num; i++)
    out[i].x = in[i];
}

template <>
inline void Data<float2>::to_float_convert(float2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}
template <>
inline void Data<double2>::to_float_convert(double2* in, float* out) {
  for (int i = 0; i < element_num; i++)
    out[i] = in[i].x;
}

bool compare_result(float* expect, float* result, int element_num) {
  for (int i = 0; i < element_num; i++) {
    if (std::abs(result[i]-expect[i]) >= 0.05) {
      return false;
    }
  }
  return true;
}

bool test_passed = true;

void test_cusolverDnTsygvd() {
  std::vector<float> a = {1, 2, 3, 2, 1, 2, 3, 2, 1};
  std::vector<float> b = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 9);
  Data<double> a_d(a.data(), 9);
  Data<float> b_s(b.data(), 9);
  Data<double> b_d(b.data(), 9);
  Data<float> w_s(3);
  Data<double> w_d(3);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  b_s.H2D();
  b_d.H2D();

  int lwork_s;
  int lwork_d;
  int status;
  status = cusolverDnSsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, &lwork_s);
  printf("status=%d\n", status);
  status = cusolverDnDsygvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, &lwork_d);
  printf("status=%d\n", status);

  float* work_s;
  double* work_d;
  int *devInfo;
  cudaMalloc(&work_s, sizeof(float) * lwork_s);
  cudaMalloc(&work_d, sizeof(double) * lwork_d);
  cudaMalloc(&devInfo, sizeof(int));


  int info_h;
  status = cusolverDnSsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, work_s, lwork_s, devInfo);
  cudaMemcpy(&info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  printf("status=%d\n", status);
  printf("info_h=%d\n", info_h);
  status = cusolverDnDsygvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, work_d, lwork_d, devInfo);
  cudaMemcpy(&info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  printf("status=%d\n", status);
  printf("info_h=%d\n", info_h);

  a_s.D2H();
  a_d.D2H();
  b_s.D2H();
  b_d.D2H();
  w_s.D2H();
  w_d.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);
  cudaFree(work_s);
  cudaFree(work_d);
  cudaFree(devInfo);

  float expect_a[9] = {-0.500000,-0.000000,0.500000,0.194937,-0.484769,0.194937,0.679705,0.874642,0.679705};
  float expect_b[9] = {1.414214,-1.000000,0.000000,-0.707107,1.224745,-1.000000,0.000000,-0.816497,1.154701};
  float expect_w[3] = {-1.000000,-0.216991,9.216990};
  if (compare_result(expect_a, a_s.h_data, 9)
      && compare_result(expect_b, b_s.h_data, 9)
      && compare_result(expect_w, w_s.h_data, 3)
      && compare_result(expect_a, a_d.h_data, 9)
      && compare_result(expect_b, b_d.h_data, 9)
      && compare_result(expect_w, w_d.h_data, 3))
    printf("DnTsygvd pass\n");
  else {
    printf("DnTsygvd fail\n");
    test_passed = false;
  }
}

void test_cusolverDnThegvd() {
  std::vector<float> a = {1, 2, 3, 2, 1, 2, 3, 2, 1};
  std::vector<float> b = {2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float2> a_s(a.data(), 9);
  Data<double2> a_d(a.data(), 9);
  Data<float2> b_s(b.data(), 9);
  Data<double2> b_d(b.data(), 9);
  Data<float> w_s(3);
  Data<double> w_d(3);

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  b_s.H2D();
  b_d.H2D();

  int lwork_s;
  int lwork_d;
  int status;
  status = cusolverDnChegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, &lwork_s);
  printf("status=%d\n", status);
  status = cusolverDnZhegvd_bufferSize(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, &lwork_d);
  printf("status=%d\n", status);

  float2* work_s;
  double2* work_d;
  int *devInfo;
  cudaMalloc(&work_s, sizeof(float2) * lwork_s);
  cudaMalloc(&work_d, sizeof(double2) * lwork_d);
  cudaMalloc(&devInfo, sizeof(int));


  int info_h;
  status = cusolverDnChegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_s.d_data, 3, b_s.d_data, 3, w_s.d_data, work_s, lwork_s, devInfo);
  cudaMemcpy(&info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  printf("status=%d\n", status);
  printf("info_h=%d\n", info_h);
  status = cusolverDnZhegvd(handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, 3, a_d.d_data, 3, b_d.d_data, 3, w_d.d_data, work_d, lwork_d, devInfo);
  cudaMemcpy(&info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  printf("status=%d\n", status);
  printf("info_h=%d\n", info_h);

  a_s.D2H();
  a_d.D2H();
  b_s.D2H();
  b_d.D2H();
  w_s.D2H();
  w_d.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);
  cudaFree(work_s);
  cudaFree(work_d);
  cudaFree(devInfo);

  float expect_a[9] = {-0.500000,-0.000000,0.500000,0.194937,-0.484769,0.194937,0.679705,0.874642,0.679705};
  float expect_b[9] = {1.414214,-1.000000,0.000000,-0.707107,1.224745,-1.000000,0.000000,-0.816497,1.154701};
  float expect_w[3] = {-1.000000,-0.216991,9.216990};
  if (compare_result(expect_a, a_s.h_data, 9)
      && compare_result(expect_b, b_s.h_data, 9)
      && compare_result(expect_w, w_s.h_data, 3)
      && compare_result(expect_a, a_d.h_data, 9)
      && compare_result(expect_b, b_d.h_data, 9)
      && compare_result(expect_w, w_d.h_data, 3))
    printf("DnThegvd pass\n");
  else {
    printf("DnThegvd fail\n");
    test_passed = false;
  }
}

struct Ptr_Data {
  int group_num;
  void** h_data;
  void** d_data;
  Ptr_Data(int group_num) : group_num(group_num) {
    h_data = (void**)malloc(group_num * sizeof(void*));
    memset(h_data, 0, group_num * sizeof(void*));
    cudaMalloc(&d_data, group_num * sizeof(void*));
    cudaMemset(d_data, 0, group_num * sizeof(void*));
  }
  ~Ptr_Data() {
    free(h_data);
    cudaFree(d_data);
  }
  void H2D() {
    cudaMemcpy(d_data, h_data, group_num * sizeof(void*), cudaMemcpyHostToDevice);
  }
};

#ifndef DPCT_USM_LEVEL_NONE
void test_cusolverDnTpotrfBatched() {
  std::vector<float> a = {2, -1, 0, -1, 2, -1, 0, -1, 2,
                          2, -1, 0, -1, 2, -1, 0, -1, 2};
  Data<float> a_s(a.data(), 18);
  Data<double> a_d(a.data(), 18);
  Data<float2> a_c(a.data(), 18);
  Data<double2> a_z(a.data(), 18);

  Ptr_Data a_s_ptrs(2); a_s_ptrs.h_data[0] = a_s.d_data; a_s_ptrs.h_data[1] = a_s.d_data + 9;
  Ptr_Data a_d_ptrs(2); a_d_ptrs.h_data[0] = a_d.d_data; a_d_ptrs.h_data[1] = a_d.d_data + 9;
  Ptr_Data a_c_ptrs(2); a_c_ptrs.h_data[0] = a_c.d_data; a_c_ptrs.h_data[1] = a_c.d_data + 9;
  Ptr_Data a_z_ptrs(2); a_z_ptrs.h_data[0] = a_z.d_data; a_z_ptrs.h_data[1] = a_z.d_data + 9;

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  a_s_ptrs.H2D();
  a_d_ptrs.H2D();
  a_c_ptrs.H2D();
  a_z_ptrs.H2D();

  int *infoArray;
  cudaMalloc(&infoArray, 2 * sizeof(int));

  int info[2];
  cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (float **)a_s_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);
  cusolverDnDpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (double **)a_d_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);
  cusolverDnCpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (float2 **)a_c_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);
  cusolverDnZpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, (double2 **)a_z_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);

  float expect[18] = { 1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701,
                       1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701 };
  if (compare_result(expect, a_s.h_data, 18) &&
      compare_result(expect, a_d.h_data, 18) &&
      compare_result(expect, a_c.h_data, 18) &&
      compare_result(expect, a_z.h_data, 18))
    printf("DnTpotrfBatched pass\n");
  else {
    printf("DnTpotrfBatched fail\n");
    test_passed = false;
  }
}

void test_cusolverDnTpotrsBatched() {
  std::vector<float> a = {1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701,
                          1.414214,-0.707107,0.000000,-0.707107,1.224745,-0.816497,0.000000,-0.816497,1.154701 };
  Data<float> a_s(a.data(), 18);
  Data<double> a_d(a.data(), 18);
  Data<float2> a_c(a.data(), 18);
  Data<double2> a_z(a.data(), 18);

  Ptr_Data a_s_ptrs(2); a_s_ptrs.h_data[0] = a_s.d_data; a_s_ptrs.h_data[1] = a_s.d_data + 9;
  Ptr_Data a_d_ptrs(2); a_d_ptrs.h_data[0] = a_d.d_data; a_d_ptrs.h_data[1] = a_d.d_data + 9;
  Ptr_Data a_c_ptrs(2); a_c_ptrs.h_data[0] = a_c.d_data; a_c_ptrs.h_data[1] = a_c.d_data + 9;
  Ptr_Data a_z_ptrs(2); a_z_ptrs.h_data[0] = a_z.d_data; a_z_ptrs.h_data[1] = a_z.d_data + 9;

  std::vector<float> b = {0, 0, 4,
                          0, 0, 4};
  Data<float> b_s(b.data(), 6);
  Data<double> b_d(b.data(), 6);
  Data<float2> b_c(b.data(), 6);
  Data<double2> b_z(b.data(), 6);

  Ptr_Data b_s_ptrs(2); b_s_ptrs.h_data[0] = b_s.d_data; b_s_ptrs.h_data[1] = b_s.d_data + 3;
  Ptr_Data b_d_ptrs(2); b_d_ptrs.h_data[0] = b_d.d_data; b_d_ptrs.h_data[1] = b_d.d_data + 3;
  Ptr_Data b_c_ptrs(2); b_c_ptrs.h_data[0] = b_c.d_data; b_c_ptrs.h_data[1] = b_c.d_data + 3;
  Ptr_Data b_z_ptrs(2); b_z_ptrs.h_data[0] = b_z.d_data; b_z_ptrs.h_data[1] = b_z.d_data + 3;

  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);

  a_s.H2D();
  a_d.H2D();
  a_c.H2D();
  a_z.H2D();

  a_s_ptrs.H2D();
  a_d_ptrs.H2D();
  a_c_ptrs.H2D();
  a_z_ptrs.H2D();

  b_s.H2D();
  b_d.H2D();
  b_c.H2D();
  b_z.H2D();

  b_s_ptrs.H2D();
  b_d_ptrs.H2D();
  b_c_ptrs.H2D();
  b_z_ptrs.H2D();

  int *infoArray;
  cudaMalloc(&infoArray, 2 * sizeof(int));

  int info[2];
  int status;
  status = cusolverDnSpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (float **)a_s_ptrs.d_data, 3, (float **)b_s_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);
  printf("status:%d\n", status);
  status = cusolverDnDpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (double **)a_d_ptrs.d_data, 3, (double **)b_s_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);
  printf("status:%d\n", status);
  status = cusolverDnCpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (float2 **)a_c_ptrs.d_data, 3, (float2 **)b_s_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);
  printf("status:%d\n", status);
  status = cusolverDnZpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, 3, 1, (double2 **)a_z_ptrs.d_data, 3, (double2 **)b_s_ptrs.d_data, 3, infoArray, 2);
  cudaMemcpy(&info, infoArray, 2*sizeof(int), cudaMemcpyDeviceToHost);
  printf("info:%d,%d\n", info[0], info[1]);
  printf("status:%d\n", status);

  a_s.D2H();
  a_d.D2H();
  a_c.D2H();
  a_z.D2H();

  b_s.D2H();
  b_d.D2H();
  b_c.D2H();
  b_z.D2H();

  cudaStreamSynchronize(0);

  cusolverDnDestroy(handle);

  float expect[6] = { 1,2,3,
                      1,2,3 };
  if (compare_result(expect, b_s.h_data, 6) &&
      compare_result(expect, b_d.h_data, 6) &&
      compare_result(expect, b_c.h_data, 6) &&
      compare_result(expect, b_z.h_data, 6))
    printf("DnTpotrsBatched pass\n");
  else {
    printf("DnTpotrsBatched fail\n");
    test_passed = false;
  }
}
#endif


int main() {
  test_cusolverDnTsygvd();
  test_cusolverDnThegvd();
#ifndef DPCT_USM_LEVEL_NONE
  //test_cusolverDnTpotrfBatched();
  //test_cusolverDnTpotrsBatched();
#endif

  if (test_passed)
    return 0;
  return -1;
}
