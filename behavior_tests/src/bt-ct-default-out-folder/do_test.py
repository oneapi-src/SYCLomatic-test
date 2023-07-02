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
    src = ["hello.cu", "hello2.cu"]

    in_root = ""
    extra_args = ""
    out_root_path = "out"
    do_migrate(src, in_root, out_root_path, single_case_text, extra_args)
    if os.path.exists(os.path.join("out", "hello.dp.cpp")):
        return True
    return False

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
