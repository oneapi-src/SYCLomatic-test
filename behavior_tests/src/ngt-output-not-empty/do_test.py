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
    os.mkdir("dpct_output")
    with open(os.path.join("dpct_output", "test"), 'w') as f:
        f.write("2")
    call_subprocess(single_case_text.CT_TOOL + " vector_add.cu  --cuda-include-path=" + single_case_text.include_path, single_case_text)
    return is_sub_string("directory is not empty.", single_case_text.print_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True