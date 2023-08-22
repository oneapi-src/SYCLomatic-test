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
    call_subprocess(test_config.CT_TOOL + " --version")
    ct_clang_version = get_ct_clang_version()
    expected_output = "dpct version {0}".format(ct_clang_version)
    print("expected dpct version output: {0}".format(expected_output))
    print("\n'dpct --version' outputs {0}".format(test_config.command_output))
    return is_sub_string(expected_output, test_config.command_output)

def build_test():
    return True

def run_test():
    return True