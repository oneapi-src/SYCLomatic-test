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
    include_path = os.path.join(os.getcwd(), "include")
    in_root = os.getcwd()
    test_case_path = os.path.join(in_root, "vector_add.cu")
    call_subprocess(test_config.CT_TOOL + " " + test_case_path + " --out-root=out --in-root=" + in_root + " --cuda-include-path=" + include_path)
    return is_sub_string("Error: The version of CUDA header files specified by --cuda-include-path is not supported. See Release Notes for supported versions.", test_config.command_output)

def build_test():
    return True
def run_test():
    return True