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
    return True

def migrate_test(single_case_text):
    change_dir('cannot-acc-dir-in-db/helloworld', single_case_text)
    call_subprocess('intercept-build /usr/bin/make', single_case_text)
    change_dir('..', single_case_text)
    call_subprocess('mv helloworld helloworld_tst', single_case_text)

    call_subprocess(test_config.CT_TOOL + ' -p ./helloworld_tst/compile_commands.json --cuda-include-path=' + \
                   os.environ['CUDA_INCLUDE_PATH'], single_case_text)

    if 'check if the directory exists and can be accessed by the tool' in single_case_text.print_text:
        return True
    print("could not get expected message: check if the directory exists and can be accessed by the tool")
    return False

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
