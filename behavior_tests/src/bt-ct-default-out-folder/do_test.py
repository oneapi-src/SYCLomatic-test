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

def setup_test():
    change_dir(test_config.current_test)

    return True



def migrate_test():
    src = ["hello.cu", "hello2.cu"]

    in_root = ""
    extra_args = ""
    out_root_path = "out"
    do_migrate(src, in_root, out_root_path, extra_args)
    if os.path.exists(os.path.join("out", "hello.dp.cpp")):
        return True
    return False

def build_test():
    return True

def run_test():
    return True
