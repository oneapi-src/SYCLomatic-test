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
    include_path = os.path.join(os.getcwd(), "include")
    in_root = os.getcwd()
    test_case_path = os.path.join(in_root, "vector_add.cu")
    call_subprocess(test_config.CT_TOOL + " " + test_case_path + " --out-root=out --in-root=" + in_root + " --cuda-include-path=" + include_path, single_case_text)
    return is_sub_string("Error: The version of CUDA header files specified by --cuda-include-path is not supported. See Release Notes for supported versions.", single_case_text.print_text)

def build_test(single_case_text):
    return True
def run_test(single_case_text):
    return True