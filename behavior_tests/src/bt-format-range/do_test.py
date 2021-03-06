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

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():

    os.mkdir("out_migrated")
    os.mkdir("out_all")
    os.mkdir("out_none")
    call_subprocess(test_config.CT_TOOL + " --format-range=migrated --out-root=./out_migrated vector_add.cu --cuda-include-path=" + test_config.include_path)
    call_subprocess(test_config.CT_TOOL + " --format-range=all --out-root=./out_all vector_add.cu  --cuda-include-path=" + test_config.include_path)
    call_subprocess(test_config.CT_TOOL + " -format-range=none --out-root=./out_none vector_add.cu --cuda-include-path=" + test_config.include_path)

    ret = is_sub_string(";\/\/ variable declearation", "./out_llvm/vector_add.dp.cpp")
    ret = is_sub_string(";\/\/ allocate device memory", "./out_google/vector_add.dp.cpp") or ret
    ret = is_sub_string(";// variable declearation", "./out_custom/vector_add.dp.cpp") or ret
    ret = is_sub_string("; // allocate device memory", "./out_custom/vector_add.dp.cpp") or ret
    ret = is_sub_string("; // variable declearation", "./out_custom/vector_add.dp.cpp") or ret
    ret = is_sub_string("; // allocate device memory", "./out_custom/vector_add.dp.cpp") or ret
    return not ret
def build_test():
    return True

def run_test():
    return True
