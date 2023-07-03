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
    max_len = 511
    if (platform.system() == 'Windows'):
        max_len = 32
    long_path = ""

    for num in range(0, max_len):
        long_path = os.path.join(long_path, "test_path")
    os.path.join(long_path, "name")
    call_subprocess("intercept-build --cdb " +
        long_path, single_case_text)
    return is_sub_string("File name", single_case_text.command_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True