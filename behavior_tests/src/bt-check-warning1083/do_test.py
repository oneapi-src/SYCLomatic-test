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
    ret_file = ""

    call_subprocess(single_case_text.CT_TOOL + " test.cu --out-root=out --cuda-include-path=" + single_case_text.include_path, single_case_text)
    ret = ""
    with open(os.path.join("out", "test.dp.cpp"), 'r') as f:
        ret = f.read()
    if not is_sub_string("1083", ret):
        return False
    change_dir("out", single_case_text)
    return True

def build_test(single_case_text):
    src_files = ["test.dp.cpp"]
    return compile_files(src_files, single_case_text)

def run_test(single_case_text):
    return True