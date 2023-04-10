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

    call_subprocess(test_config.CT_TOOL + " --optimize-migration --out-root=./out kernel-func.cu --cuda-include-path=" + test_config.include_path)

    ret = is_sub_string("Recursive functions cannot be called", test_config.command_output)
    ret = is_sub_string("Virtual functions cannot be called in a SYCL kernel", test_config.command_output) and ret
    return ret
def build_test():
    return True

def run_test():
    return True
