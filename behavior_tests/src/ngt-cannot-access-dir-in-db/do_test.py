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
    cur_dr = os.getcwd()
    ret = ""
    with open("compile_commands.json", 'r') as f:
        ret = f.read()
    ret = ret.replace("directory_placeholder", cur_dr)
    with open("compile_commands.json", 'w') as f:
        f.write(ret)

    call_subprocess(test_config.CT_TOOL + " simple_foo.cu --cuda-include-path=" + test_config.include_path)

    return is_sub_string("Error: Cannot access directory", test_config.command_output)

def build_test():
    return True

def run_test():
    return True