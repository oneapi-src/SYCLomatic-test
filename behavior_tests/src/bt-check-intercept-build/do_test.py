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
from hashlib import md5

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    # Add dpct bin dir to ENV PATH viriable
    dpct_bin_dir = os.path.join(os.path.dirname(shutil.which("dpct")), '..')
    os.environ["PATH"] += os.pathsep + dpct_bin_dir

    # clean previous migration output
    if (os.path.exists("out")):
        shutil.rmtree("out")

    ret = call_subprocess("intercept-build /usr/bin/make -B")
    if not ret:
        print("Error to create compilation database:", test_config.command_output)
    ret = call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " -in-root=.  -out-root=./out --gen-build-script -p ./")
    if not ret:
        print("Error to migration:", test_config.command_output)
    ret = call_subprocess("cd out && make -f Makefile.dpct")
    if not ret:
        print("Error to build:", test_config.command_output)
    ret =  os.path.exists("out/main.dp.o")
    return ret

def build_test():
    return True

def run_test():
    return True

