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

    call_subprocess(test_config.CT_TOOL + " simple_foo.cu -stop-on-parse-err --cuda-include-path=" + test_config.include_path)
    return (is_sub_string("Cannot parse input file", test_config.command_output)
            and is_sub_string("Parsing", test_config.command_output)
            and not is_sub_string("Analyzing", test_config.command_output)
            and not is_sub_string("Migrating", test_config.command_output))

def build_test():
    return True

def run_test():
    return True