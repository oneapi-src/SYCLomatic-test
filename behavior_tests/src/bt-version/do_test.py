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
    call_subprocess(test_config.CT_TOOL + " --version", single_case_text)
    ct_clang_version = get_ct_clang_version()
    expected_output = "dpct version {0}".format(ct_clang_version)
    print("expected dpct version output: {0}".format(expected_output))
    print("\n'dpct --version' outputs {0}".format(single_case_text.print_text))
    return is_sub_string(expected_output, single_case_text.print_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True