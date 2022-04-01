// ====------ macro.cu---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CONCAT(name) cuda##name

int eventSynchronize(cudaEvent_t event) {
  return CONCAT(EventSynchronize)(event);
}

typedef CONCAT(Stream_t) stream_t;
typedef CONCAT(Event_t) event_t;


inline int streamCreate(stream_t *stream) {
  return CONCAT(StreamCreate)(stream);
}

inline int streamCreateWithFlags(stream_t *stream, unsigned int flags) {
  return CONCAT(StreamCreateWithFlags)(stream, flags);
}

inline int streamCreateWithPriority(stream_t *stream, unsigned int flags,
                                    int priority) {
  return CONCAT(StreamCreateWithPriority)(stream, flags, priority);
}

inline int streamDestroy(stream_t stream) {
  return CONCAT(StreamDestroy)(stream);
}

int foo(int num) {
#if CUDART_VERSION >= 4000
  cudaDeviceReset();
#else
  cudaThreadExit();
#endif
}

