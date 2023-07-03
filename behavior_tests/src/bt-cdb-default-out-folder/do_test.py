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
    call_subprocess("intercept-build /usr/bin/make", single_case_text)
    in_root = ""
    extra_args = ""
    return call_subprocess(single_case_text.CT_TOOL + " --cuda-include-path=" + single_case_text.include_path + " " +
        "-p .", single_case_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
