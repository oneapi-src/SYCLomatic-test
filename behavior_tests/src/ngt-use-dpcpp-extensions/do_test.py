# ====------ do_test.py---------- *- Python -* ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#
import subprocess
import platform
import os
import sys
from test_config import CT_TOOL

from test_utils import *

def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    return True

def migrate_test(single_case_text):

    call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path +
        " --use-dpcpp-extensions=abc", single_case_text)
    return is_sub_string("Cannot find option named 'abc'", single_case_text.print_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True