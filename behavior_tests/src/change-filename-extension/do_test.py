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


def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    return True


def migrate_test(single_case_text):
    call_subprocess(
        test_config.CT_TOOL + " -p=. --change-cuda-files-extension-only --out-root=out --cuda-include-path=" + test_config.include_path, single_case_text)
    print(single_case_text.command_text)

    reference = 'main.dp.cpp'
    call_subprocess("ls out | grep " + reference, single_case_text)
    res = True
    if reference not in single_case_text.command_text:
        res = False
        print("there should be a file: " + reference)

    reference = 'test.cpp'
    call_subprocess("ls out | grep " + reference, single_case_text)
    if reference not in single_case_text.command_text:
        res = False
        print("there should be a file: " + reference)

    reference = 'test.dp.hpp'
    call_subprocess("ls out | grep " + reference, single_case_text)
    if reference not in single_case_text.command_text:
        res = False
        print("there should be a file: " + reference)

    reference = 'test.h'
    call_subprocess("ls out | grep " + reference, single_case_text)
    if reference not in single_case_text.command_text:
        res = False
        print("there should be a file: " + reference)

    return res


def build_test(single_case_text):
    return True


def run_test(single_case_text):
    return True
