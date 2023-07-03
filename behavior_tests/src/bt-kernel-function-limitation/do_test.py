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

    call_subprocess(single_case_text.CT_TOOL + " --optimize-migration --out-root=./out kernel-func.cu --cuda-include-path=" + single_case_text.include_path, single_case_text)

    ret = is_sub_string("Recursive functions cannot be called in SYCL device code", single_case_text.print_text)
    ret = is_sub_string("Virtual functions cannot be called in SYCL device code", single_case_text.print_text) and ret
    return ret
def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
