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
from test_config import CT_TOOL

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    call_subprocess("intercept-build /usr/bin/make")
    in_root = ""
    extra_args = ""
    call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " " +
        "-p .")
    return False
def build_test():
    return True

def run_test():
    return True