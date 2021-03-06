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
    call_subprocess("mkdir ./read_only_folder")
    call_subprocess("chmod 0444 ./read_only_folder")
    call_subprocess(test_config.CT_TOOL + " simple_foo.cu --out-root=./read_only_folder --cuda-include-path=" + test_config.include_path)
    return is_sub_string("Unable to save the output to the specified directory", test_config.command_output)
def build_test():
    call_subprocess("rm -rf ./read_only_folder")
    call_subprocess("chmod 0777 ./read_only_folder")
    return True

def run_test():
    return True