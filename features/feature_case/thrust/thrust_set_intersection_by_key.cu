// ====------ set_intersection_by_key.cu-------- *- CUDA -*---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

void test_1() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};
  thrust::host_vector<int> VAkey(Akey, Akey + N);
  thrust::host_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::host_vector<int> VBkey(Bkey, Bkey + M);

  thrust::host_vector<int> VCkey(Ckey, Ckey + P);
  thrust::host_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      thrust::host, VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(),
      VAvalue.begin(), VCkey.begin(), VCvalue.begin());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_1 run failed\n");
      exit(-1);
    }
  }

  printf("test_1 run passed!\n");
}

void test_2() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};
  thrust::host_vector<int> VAkey(Akey, Akey + N);
  thrust::host_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::host_vector<int> VBkey(Bkey, Bkey + M);

  thrust::host_vector<int> VCkey(Ckey, Ckey + P);
  thrust::host_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(), VAvalue.begin(),
      VCkey.begin(), VCvalue.begin());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_2 run failed\n");
      exit(-1);
    }
  }

  printf("test_2 run passed!\n");
}

void test_3() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {11, 9, 7, 5, 3, 1};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {13, 8, 5, 3, 2, 1, 1};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {5, 3, 1};
  int ansvalue[P] = {0, 0, 0};
  thrust::host_vector<int> VAkey(Akey, Akey + N);
  thrust::host_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::host_vector<int> VBkey(Bkey, Bkey + M);

  thrust::host_vector<int> VCkey(Ckey, Ckey + P);
  thrust::host_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      thrust::host, VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(),
      VAvalue.begin(), VCkey.begin(), VCvalue.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_3 run failed\n");
      exit(-1);
    }
  }

  printf("test_3 run passed!\n");
}

void test_4() {

  const int N = 6, M = 7, P = 3;
  int Akey[N] = {11, 9, 7, 5, 3, 1};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {13, 8, 5, 3, 2, 1, 1};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {5, 3, 1};
  int ansvalue[P] = {0, 0, 0};
  thrust::host_vector<int> VAkey(Akey, Akey + N);
  thrust::host_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::host_vector<int> VBkey(Bkey, Bkey + M);

  thrust::host_vector<int> VCkey(Ckey, Ckey + P);
  thrust::host_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::host_vector<int>::iterator,
                       thrust::host_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(), VAvalue.begin(),
      VCkey.begin(), VCvalue.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_4 run failed\n");
      exit(-1);
    }
  }

  printf("test_4 run passed!\n");
}

void test_5() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};
  thrust::device_vector<int> VAkey(Akey, Akey + N);
  thrust::device_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::device_vector<int> VBkey(Bkey, Bkey + M);

  thrust::device_vector<int> VCkey(Ckey, Ckey + P);
  thrust::device_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      thrust::device, VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(),
      VAvalue.begin(), VCkey.begin(), VCvalue.begin());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_5 run failed\n");
      exit(-1);
    }
  }

  printf("test_5 run passed!\n");
}

void test_6() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};
  thrust::device_vector<int> VAkey(Akey, Akey + N);
  thrust::device_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::device_vector<int> VBkey(Bkey, Bkey + M);

  thrust::device_vector<int> VCkey(Ckey, Ckey + P);
  thrust::device_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(), VAvalue.begin(),
      VCkey.begin(), VCvalue.begin());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_6 run failed\n");
      exit(-1);
    }
  }

  printf("test_6 run passed!\n");
}

void test_7() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {11, 9, 7, 5, 3, 1};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {13, 8, 5, 3, 2, 1, 1};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {5, 3, 1};
  int ansvalue[P] = {0, 0, 0};
  thrust::device_vector<int> VAkey(Akey, Akey + N);
  thrust::device_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::device_vector<int> VBkey(Bkey, Bkey + M);

  thrust::device_vector<int> VCkey(Ckey, Ckey + P);
  thrust::device_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      thrust::device, VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(),
      VAvalue.begin(), VCkey.begin(), VCvalue.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_7 run failed\n");
      exit(-1);
    }
  }

  printf("test_7 run passed!\n");
}

void test_8() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {11, 9, 7, 5, 3, 1};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {13, 8, 5, 3, 2, 1, 1};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {5, 3, 1};
  int ansvalue[P] = {0, 0, 0};
  thrust::device_vector<int> VAkey(Akey, Akey + N);
  thrust::device_vector<int> VAvalue(Avalue, Avalue + N);

  thrust::device_vector<int> VBkey(Bkey, Bkey + M);

  thrust::device_vector<int> VCkey(Ckey, Ckey + P);
  thrust::device_vector<int> VCvalue(Cvalue, Cvalue + P);
  typedef thrust::pair<thrust::device_vector<int>::iterator,
                       thrust::device_vector<int>::iterator>
      iter_pair;

  iter_pair end = thrust::set_intersection_by_key(
      VAkey.begin(), VAkey.end(), VBkey.begin(), VBkey.end(), VAvalue.begin(),
      VCkey.begin(), VCvalue.begin(), thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (VCkey[i] != anskey[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (VCvalue[i] != ansvalue[i]) {
      printf("test_8 run failed\n");
      exit(-1);
    }
  }

  printf("test_8 run passed!\n");
}

void test_9() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};

  thrust::set_intersection_by_key(
      thrust::host, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
  for (int i = 0; i < P; i++) {
    if (Ckey[i] != anskey[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (Cvalue[i] != ansvalue[i]) {
      printf("test_9 run failed\n");
      exit(-1);
    }
  }

  printf("test_9 run passed!\n");
}

void test_10() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {1, 3, 5, 7, 9, 11};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {1, 1, 2, 3, 5, 8, 13};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {1, 3, 5};
  int ansvalue[P] = {0, 0, 0};

  thrust::set_intersection_by_key(
      Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue);
  for (int i = 0; i < P; i++) {
    if (Ckey[i] != anskey[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (Cvalue[i] != ansvalue[i]) {
      printf("test_10 run failed\n");
      exit(-1);
    }
  }

  printf("test_10 run passed!\n");
}

void test_11() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {11, 9, 7, 5, 3, 1};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {13, 8, 5, 3, 2, 1, 1};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {5, 3, 1};
  int ansvalue[P] = {0, 0, 0};

  thrust::set_intersection_by_key(
      thrust::host, Akey, Akey + N, Bkey, Bkey + M, Avalue, Ckey, Cvalue,
      thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (Ckey[i] != anskey[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (Cvalue[i] != ansvalue[i]) {
      printf("test_11 run failed\n");
      exit(-1);
    }
  }

  printf("test_11 run passed!\n");
}

void test_12() {
  const int N = 6, M = 7, P = 3;
  int Akey[N] = {11, 9, 7, 5, 3, 1};
  int Avalue[N] = {0, 0, 0, 0, 0, 0};
  int Bkey[M] = {13, 8, 5, 3, 2, 1, 1};

  int Ckey[P];
  int Cvalue[P];
  int anskey[P] = {5, 3, 1};
  int ansvalue[P] = {0, 0, 0};

  thrust::set_intersection_by_key(Akey, Akey + N, Bkey, Bkey + M, Avalue,
                                      Ckey, Cvalue, thrust::greater<int>());
  for (int i = 0; i < P; i++) {
    if (Ckey[i] != anskey[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }
  for (int i = 0; i < P; i++) {
    if (Cvalue[i] != ansvalue[i]) {
      printf("test_12 run failed\n");
      exit(-1);
    }
  }

  printf("test_12 run passed!\n");
}

int main() {
  test_1();
  test_2();
  test_3();
  test_4();
  test_5();
  test_6();
  test_7();
  test_8();

  test_9();
  test_10();
  test_11();
  test_12();

  return 0;
}