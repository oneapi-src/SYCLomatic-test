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
    change_dir('process_optf_option/use_optf_option_case')
    call_subprocess('intercept-build /usr/bin/make')
    call_subprocess(test_config.CT_TOOL + ' -in-root ./ -out-root out -gen-build-script -p ./compile_commands.json --cuda-include-path=' + \
                   os.environ['CUDA_INCLUDE_PATH'])
    change_dir("./out")
    call_subprocess("make -f Makefile.dpct -B")

   
    if os.path.isfile("./test1.dp.o") and os.path.isfile("./test1.dp.o"):
        print("process_optf_option migrate pass")
        return True
    else:
        print("process_optf_option migrate failed")
        return False

    
def build_test():
    return True

def run_test():
    return True
