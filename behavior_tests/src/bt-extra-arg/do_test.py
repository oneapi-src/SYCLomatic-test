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
    return (call_subprocess(test_config.CT_TOOL + " --extra-arg=\"--cuda-path=" + test_config.include_path + "\" vector_add.cu")
            and is_sub_string("Parsing", test_config.command_output)
            and is_sub_string("Analyzing", test_config.command_output)
            and is_sub_string("Migrating", test_config.command_output))

def build_test():
    return True

def run_test():
    return True
