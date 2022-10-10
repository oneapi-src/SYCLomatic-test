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

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    ret = call_subprocess(test_config.CT_TOOL + " --use-explicit-namespace=cl --out-root=./sycl vector_add.cu --cuda-include-path=" + test_config.include_path)
    return ret
def build_test():
    return True

def run_test():
    os.environ["SYCL_DEVICE_FILTER"] = test_config.device_filter
    return True
