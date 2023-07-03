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
    change_dir("helloworld", single_case_text)
    return True

def migrate_test(single_case_text):
    call_subprocess("intercept-build /usr/bin/make", single_case_text)
    change_dir("..", single_case_text)
    call_subprocess("mv helloworld helloworld_tst", single_case_text)
    call_subprocess(test_config.CT_TOOL + " helloworld_tst/src/test.cu --cuda-include-path=" + test_config.include_path +
             " --suppress-warnings=-1,0,0x100,0x1000,0x3fffffff,0x7ffffffe,0x7fffffff,0x80000000,0xfffffffe,0xffffffff,0x10000,0x100000", single_case_text)

    return is_sub_string("Error: Invalid warning ID or range; valid warning IDs range from 1000 to", single_case_text.command_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True