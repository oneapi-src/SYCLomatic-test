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
    return True

def migrate_test():
    change_dir('cannot-acc-dir-in-db/helloworld')
    call_subprocess('intercept-build /usr/bin/make')
    change_dir('..')
    call_subprocess('mv helloworld helloworld_tst')

    call_subprocess(test_config.CT_TOOL + ' -p ./helloworld_tst/compile_commands.json helloworld_tst/src/test.cu --cuda-include-path=' + \
                   os.environ['CUDA_INCLUDE_PATH'])

    if 'check if the directory exists and can be accessed by the tool' in test_config.command_output:
        return True
    print("could not get expected message: check if the directory exists and can be accessed by the tool")
    return False

def build_test():
    return True

def run_test():
    return True