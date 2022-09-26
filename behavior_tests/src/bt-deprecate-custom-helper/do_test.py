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


def check_deprecation_dpct_output():
    call_subprocess(test_config.CT_TOOL +
                    " test.cu --use-custom-helper=api --custom-helper-name=my_proj --out-root=out --cuda-include-path=" + test_config.include_path)
    if not is_sub_string("Note: Option --use-custom-helper is deprecated and may be removed in the future.", test_config.command_output):
        return False
    if not is_sub_string("Note: Option --custom-helper-name is deprecated and may be removed in the future.", test_config.command_output):
        return False
    return True

def check_deprecation_dpct_help():
    call_subprocess(test_config.CT_TOOL + " --help")
    if not is_sub_string("DEPRECATED: Specifies the helper headers folder name and main helper header file name.", test_config.command_output):
        return False
    if not is_sub_string("DEPRECATED: Customize the helper header files for migrated code.", test_config.command_output):
        return False
    return True

def migrate_test():
    return check_deprecation_dpct_output() and check_deprecation_dpct_help()

def build_test():
    return True


def run_test():
    return True
