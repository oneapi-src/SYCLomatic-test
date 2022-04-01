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
    change_dir("helloworld")
    return True

def migrate_test():
    call_subprocess("intercept-build /usr/bin/make")
    change_dir("..")
    call_subprocess("mv helloworld helloworld_tst")
    call_subprocess(test_config.CT_TOOL + " helloworld_tst/src/test.cu --cuda-include-path=" + test_config.include_path +
             " --suppress-warnings=-1,0,0x100,0x1000,0x3fffffff,0x7ffffffe,0x7fffffff,0x80000000,0xfffffffe,0xffffffff,0x10000,0x100000")

    return is_sub_string("Error: Invalid warning ID or range; valid warning IDs range from 1000 to", test_config.command_output)

def build_test():
    return True

def run_test():
    return True