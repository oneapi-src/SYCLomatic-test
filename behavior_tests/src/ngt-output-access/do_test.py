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
    call_subprocess("chmod u-r dpct_output", single_case_text)
    call_subprocess("dpct vector_add.cu --cuda-include-path=" + single_case_text.include_path, single_case_text)
    return is_sub_string("Could not access output directory", single_case_text.print_text)

def build_test(single_case_text):
    call_subprocess("chmod 777 dpct_output", single_case_text)
    return True

def run_test(single_case_text):
    return True