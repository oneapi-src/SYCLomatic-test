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
    max_len = 511
    if (platform.system() == 'Windows'):
        max_len = 32
    long_path = ""

    for num in range(0, max_len):
        long_path = os.path.join(long_path, "test_path")
    os.path.join(long_path, "name")
    call_subprocess("intercept-build --cdb " +
        long_path)
    return is_sub_string("File name", test_config.command_output)

def build_test():
    return True

def run_test():
    return True