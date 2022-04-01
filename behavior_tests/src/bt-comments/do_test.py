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
    ret_file = ""
    call_subprocess(test_config.CT_TOOL + " --comments comments.cu --out-root=out --cuda-include-path=" + test_config.include_path)
    with open(os.path.join("out", "comments.dp.cpp"), 'r') as f:
        ret_file = f.read()
    return is_sub_string("//", ret_file)

def build_test():
    return True

def run_test():
    return True