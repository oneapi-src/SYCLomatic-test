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
    call_subprocess(
        test_config.CT_TOOL + " -p=. --change-cuda-files-extension-only --out-root=out --cuda-include-path=" + test_config.include_path)
    print(test_config.command_output)

    reference = 'main.dp.cpp'
    call_subprocess("ls out | grep " + reference)
    res = True
    if reference not in test_config.command_output:
        res = False
        print("there should be a file: " + reference)

    reference = 'test.cpp'
    call_subprocess("ls out | grep " + reference)
    if reference not in test_config.command_output:
        res = False
        print("there should be a file: " + reference)

    reference = 'test.dp.hpp'
    call_subprocess("ls out | grep " + reference)
    if reference not in test_config.command_output:
        res = False
        print("there should be a file: " + reference)

    reference = 'test.h'
    call_subprocess("ls out | grep " + reference)
    if reference not in test_config.command_output:
        res = False
        print("there should be a file: " + reference)

    return res


def build_test():
    return True


def run_test():
    return True
