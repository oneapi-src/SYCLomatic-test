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

def setup_test():
    change_dir(test_config.current_test)
    prepare_execution_folder()
    return True


def prepare_execution_folder():
    distutils.dir_util.copy_tree(test_config.include_path, "include")

def migrate_test():
    src =os.path.join("include", "vector_types.h")
    in_root = ""
    extra_args = ""
    call_subprocess(test_config.CT_TOOL + " --cuda-include-path=./include " + src)
    print("hello" + test_config.command_output)
    if ('option is in the CUDA_PATH folder' in test_config.command_output):
        return True
    return False

def build_test():
    return True

def run_test():
    return True