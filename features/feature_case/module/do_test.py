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
    if (os.path.exists("dpct_output")):
        shutil.rmtree("dpct_output")    
    call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " module-helper.cpp module-main.cu module-kernel.cu --extra-arg=--ptx")
    return os.path.exists(os.path.join("dpct_output", "module-kernel.dp.cpp"))
def build_test():
    # make shared library
    if (platform.system() == 'Windows'):
        ret = call_subprocess("icpx -fsycl              dpct_output/module-kernel.dp.cpp       -shared -o module-kernel.dll")
    else:
        ret = call_subprocess(test_config.DPCXX_COM + " dpct_output/module-kernel.dp.cpp -fPIC -shared -o module-kernel.so")
    if not ret:
        print("Could not make module-kernel.* shared library.")
        return False

    srcs = []
    srcs.append(os.path.join("dpct_output", "module-helper.cpp"))
    srcs.append(os.path.join("dpct_output", "module-main.dp.cpp"))
    return compile_and_link(srcs)
def run_test():
    return call_subprocess(os.path.join(os.path.curdir, test_config.current_test + '.run '))
