# ====------ do_test.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import subprocess
import platform
import os
import sys

from test_utils import *

def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    return True

def migrate_test(single_case_text):
    ret = call_subprocess(test_config.CT_TOOL + " --use-explicit-namespace=sycl-math --out-root=./sycl vector_add.cu --cuda-include-path=" + test_config.include_path, single_case_text)
    return ret
def build_test(single_case_text):
    srcs = []
    srcs.append(os.path.join("sycl", "vector_add.dp.cpp"))
    return compile_and_link(srcs, single_case_text)
    

def run_test(single_case_text):
    os.environ["ONEAPI_DEVICE_SELECTOR"] = test_config.device_filter
    return run_binary_with_args(single_case_text)