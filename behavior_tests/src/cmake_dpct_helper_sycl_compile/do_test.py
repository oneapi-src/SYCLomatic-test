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
from test_config import CT_TOOL

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    # clean previous migration output
    if (os.path.exists("build")):
        shutil.rmtree("build")

    ret = call_subprocess("mkdir build")
    if not ret:
        print("Failed to create build folder:", test_config.command_output)

    ret = change_dir("build")
    if not ret:
        print("Failed to change CWD to build folder:", test_config.command_output)

    ret = call_subprocess("cmake -G \"Unix Makefiles\" -DCMAKE_CXX_COMPILER=icpx ../")
    if not ret:
        print("Command 'cmake configure' failed:", test_config.command_output)

    ret = call_subprocess("make")
    if not ret:
        print("Command 'make' failed:", test_config.command_output)

    return os.path.exists("test_dpct_helper_sycl_compile ")

def build_test():
    return True

def run_test():
    return call_subprocess("test_dpct_helper_sycl_compile ")
