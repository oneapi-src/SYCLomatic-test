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
    call_subprocess(
        test_config.CT_TOOL + " test.cu --out-root=out --cuda-include-path=" + test_config.include_path)

    with open(os.path.join("out", "test.dp.cpp"), 'r') as f:
        ret_str = f.read()
    res = True

    reference_list = ["'S' expression", "'x' expression", "'sizeof(C) * 3' expression", "'sizeof(x) * 3' expression",
                      "'sizeof(x * 3) * 3' expression", "'S+1+S' expression"]
    for reference in reference_list:
        if reference not in ret_str:
            res = False
            print("there should be a DPCT1101 " + reference + " warning.")

    return res


def build_test(single_case_text):
    return True


def run_test(single_case_text):
    return True
