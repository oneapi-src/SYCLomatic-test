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
    # clean previous migration output
    if (os.path.exists("dpct_output")):
        shutil.rmtree("dpct_output")    
    call_subprocess(single_case_text.CT_TOOL + " --cuda-include-path=" + single_case_text.include_path + " kernel_library.cpp jit.cu --extra-arg=--ptx", single_case_text)
    return os.path.exists(os.path.join("dpct_output", "kernel_library.cpp.dp.cpp"))
def build_test(single_case_text):
    # make shared library
    if (platform.system() == 'Windows'):
        ret = call_subprocess("icpx -fsycl              dpct_output/jit.dp.cpp       -shared -o premade.ptx", single_case_text)
    else:
        ret = call_subprocess(single_case_text.DPCXX_COM + " dpct_output/jit.dp.cpp -fPIC -shared -o premade.ptx", single_case_text)
    if not ret:
        print("Could not make premade.ptx shared library.")
        return False

    srcs = []
    srcs.append(os.path.join("dpct_output", "kernel_library.cpp.dp.cpp"))
    return compile_and_link(srcs, single_case_text, linkopt=["-lstdc++fs"])
def run_test(single_case_text):
    return call_subprocess(os.path.join(os.path.curdir, single_case_text.name + '.run '), single_case_text)
