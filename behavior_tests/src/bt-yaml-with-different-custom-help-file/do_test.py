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
    if (platform.system() == 'Windows'):
        call_subprocess("where " + test_config.CT_TOOL)
        print(test_config.command_output)
    call_subprocess(test_config.CT_TOOL + " test.cu --use-custom-helper=file --custom-helper-name=my_proj2  --out-root=out --cuda-include-path=" + test_config.include_path)
    return is_sub_string("Incremental migration requires the same option sets used across different", test_config.command_output)

def build_test():
    return True

def run_test():
    return True