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
    yml_file = os.path.join("out", "test.h.yaml")
    test_file = os.path.join("out", "test.h")
    ret_str = ""
    ret = []
    call_subprocess(test_config.CT_TOOL + " test.cu --out-root=out --extra-arg=\"-xc\" --cuda-include-path=" + test_config.include_path, single_case_text)
    if not os.path.exists(yml_file):
        return False
    with open(yml_file, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.replace("in device code was not detected", "in device code was not detected!!!")
            ret.append(line)
    with open(yml_file, 'w') as f:
        f.writelines(ret)
    call_subprocess(test_config.CT_TOOL + " test.cu --out-root=out --extra-arg=\"-xcuda\" --cuda-include-path=" + test_config.include_path, single_case_text)
    with open(test_file, 'r') as f:
        ret_str = f.read()
    if ret_str.count("DPCT1056") == 1:
        return True
    return False

def build_test(single_case_text):
    return True
def run_test(single_case_text):
    return True