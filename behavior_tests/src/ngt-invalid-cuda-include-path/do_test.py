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
    call_subprocess(single_case_text.CT_TOOL +
        " --extra-arg=\"--cuda-path=/usr/local/folder-does-not-contain-cuda\" vector_add.cu", single_case_text)
    return is_sub_string("Could not detect path to CUDA header files", single_case_text.print_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True