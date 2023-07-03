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
from test_config import CT_TOOL

from test_utils import *

def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    return True

def migrate_test(single_case_text):
    data = []
    ret = []
    iter = 0
    with open("compile_commands.json", 'r') as f:
        data = f.readlines()
        for line in data:
            if iter == 1:
                line = line.replace("directory_placeholder", os.getcwd().replace("\\", "\\\\"))
            if "directory_placeholder" in line:
                iter += 1

            ret.append(line)
    with open("compile_commands.json", 'w') as f:
        f.writelines(ret)

    call_subprocess(test_config.CT_TOOL + ' -p=./ --cuda-include-path=' + test_config.include_path, single_case_text)
    return is_sub_string("Processed 1 file(s)", single_case_text.command_text)


def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True


