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

    call_subprocess("sed 's/main/scale_main/' cudnn-scale.cu --in-place")
    call_subprocess("sed 's/main/sum_main/'   cudnn-sum.cu   --in-place")    
    call_subprocess(test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " multiple_main.cpp cudnn-scale.cu cudnn-sum.cu")
    return os.path.exists(os.path.join("dpct_output", "cudnn-scale.dp.cpp"))

def build_test():
    srcs = []
    objects = []
    if platform.system() == 'Linux':    
        linkopt = ["-ldnnl"]
    else:
        linkopt = ["dnnl.lib"]
        
    srcs.append(os.path.join("dpct_output", "multiple_main.cpp"))
    srcs.append(os.path.join("dpct_output", "cudnn-scale.dp.cpp"))
    srcs.append(os.path.join("dpct_output", "cudnn-sum.dp.cpp"))
    return compile_and_link(srcs,objects,linkopt)

def run_test():
    return call_subprocess(os.path.join(os.path.curdir, test_config.current_test + '.run '))
