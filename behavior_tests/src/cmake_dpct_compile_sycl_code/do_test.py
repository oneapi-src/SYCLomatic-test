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
    return True

def migrate_test():
    # clean previous migration output
    if (os.path.exists("build")):
        shutil.rmtree("build")

    call_subprocess("mkdir build")
    change_dir("build")
    call_subprocess("cmake -G \"Unix Makefiles\" -DCMAKE_CXX_COMPILER=icpx ../")
    print("test_config.command_output 00 :", test_config.command_output)
    call_subprocess("make")
    print("test_config.command_output 11 :", test_config.command_output)
    return os.path.exists("app.run")
def build_test():
    return True
def run_test():
    return call_subprocess("./app.run")
