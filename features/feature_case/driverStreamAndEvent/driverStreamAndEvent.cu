// ====------ driverStreamAndEvent.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
int main(){
  CUfunction f;
  CUstream s;
  CUevent e;

  cuFuncSetCacheConfig(f, CU_FUNC_CACHE_PREFER_NONE);

  cuStreamCreate(&s, CU_STREAM_DEFAULT);
  cuStreamSynchronize(s);

  cuEventCreate(&e, CU_EVENT_DEFAULT);
  cuStreamWaitEvent(s, e, 0);

  cuEventRecord(e, s);
  cuEventSynchronize(e);

  CUresult r;
  r = cuEventQuery(e);

  CUevent start, end;
  cuEventRecord(start, s);
  cuEventRecord(end, s);
  cuEventSynchronize(start);
  cuEventSynchronize(end);
  float result_time;
  cuEventElapsedTime(&result_time, start, end);

  int rr;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);

  return 0;
}
