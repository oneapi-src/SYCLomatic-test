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
    # clean previous migration output
    if (os.path.exists("dpct_output")):
        shutil.rmtree("dpct_output")

    call_subprocess("sed 's/main/scale_main/' cudnn-scale.cu --in-place", single_case_text)
    call_subprocess("sed 's/main/sum_main/'   cudnn-sum.cu   --in-place", single_case_text)    
    call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " multiple_main.cpp cudnn-scale.cu cudnn-sum.cu", single_case_text)
    return os.path.exists(os.path.join("dpct_output", "cudnn-scale.dp.cpp"))

def build_test(single_case_text):
    srcs = []
    objects = []
    cmp_options = []


    cmp_options.append("-DMKL_ILP64")
    if platform.system() == 'Linux':
        linkopt = test_config.mkl_link_opt_lin
        linkopt.append(" -ldnnl")
    else:
        linkopt = test_config.mkl_link_opt_win
        linkopt.append(" dnnl.lib")

    srcs.append(os.path.join("dpct_output", "multiple_main.cpp"))
    srcs.append(os.path.join("dpct_output", "cudnn-scale.dp.cpp"))
    srcs.append(os.path.join("dpct_output", "cudnn-sum.dp.cpp"))
    return compile_and_link(srcs, single_case_text, cmp_options, objects, linkopt)

def run_test(single_case_text):
    return call_subprocess(os.path.join(os.path.curdir, single_case_text.name + '.run '), single_case_text)
