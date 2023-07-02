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
    prepare_execution_folder()
    return True


def prepare_execution_folder():
    distutils.dir_util.copy_tree(test_config.include_path, "include")

def migrate_test(single_case_text):
    src =os.path.join("include", "vector_types.h")
    in_root = ""
    extra_args = ""
    call_subprocess(test_config.CT_TOOL + " --cuda-include-path=./include " + src)
    print("hello" + single_case_text.command_text)
    if ('option is in the CUDA_PATH folder' in single_case_text.command_text):
        return True
    return False

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True