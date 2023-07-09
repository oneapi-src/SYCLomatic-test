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
    call_subprocess("mkdir ./read_only_folder", single_case_text)
    call_subprocess("chmod 0444 ./read_only_folder", single_case_text)
    call_subprocess(single_case_text.CT_TOOL + " simple_foo.cu --out-root=./read_only_folder --cuda-include-path=" + single_case_text.include_path, single_case_text)
    return is_sub_string("Unable to save the output to the specified directory", single_case_text.print_text)
def build_test(single_case_text):
    call_subprocess("rm -rf ./read_only_folder", single_case_text)
    call_subprocess("chmod 0777 ./read_only_folder", single_case_text)
    return True

def run_test(single_case_text):
    return True