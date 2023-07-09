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
    print("test-----------------")
    call_subprocess(single_case_text.CT_TOOL + ' hello.h --out-root=out --cuda-include-path=' + \
                   os.environ['CUDA_INCLUDE_PATH'], single_case_text)
    file_data=""
    with open('out/hello.h.yaml', 'r') as f:
        file_data = f.read()

    file_data = file_data.replace("- FilePath", "FilePath",1)
    print(file_data)
    with open('out/hello.h.yaml', 'w') as f:
        f.write(file_data)

    call_subprocess(single_case_text.CT_TOOL + ' hello.h --out-root=./out --cuda-include-path=' + \
                   os.environ['CUDA_INCLUDE_PATH'], single_case_text)
    print(single_case_text.print_text)
    if "Unexpected token. Expected Key or Block End" in single_case_text.print_text:
        return True
    print("not catch the error: unkown key constantFla")
    return False


def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
