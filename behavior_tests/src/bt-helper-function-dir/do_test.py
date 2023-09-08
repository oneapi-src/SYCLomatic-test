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
    call_subprocess(test_config.CT_TOOL + " --helper-function-dir")
    helper_function_dir_root = os.path.realpath(
        os.path.join(get_ct_path(), os.pardir, os.pardir, "include"))
    helper_function_cmd_output = test_config.command_output.replace("\n","")
    print("Helper function directory: ", helper_function_dir_root)
    print("Helper function command output: ", helper_function_cmd_output)
    return os.path.samefile(helper_function_dir_root, helper_function_cmd_output)

def build_test():
    return True

def run_test():
    return True
