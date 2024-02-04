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
    call_subprocess(test_config.CT_TOOL + " vector_add.cu --out-root=/home/root --cuda-include-path=" + test_config.include_path)
    if not is_sub_string("[ERROR] Create Directory : /home/root fail: Permission denied", test_config.command_output):
        print('Cannot find the message "[ERROR] Create Directory : /home/root fail: Permission denied"')
        return False
    call_subprocess("mkdir test_out")
    call_subprocess("chmod 555 test_out")
    call_subprocess(test_config.CT_TOOL + " vector_add.cu --out-root=test_out --cuda-include-path=" + test_config.include_path)
    if not is_sub_string("unable to overwrite file", test_config.command_output):
        print('Cannot find the message "unable to overwrite file"')
        return False
    return True

def build_test():
    return True

def run_test():
    return True