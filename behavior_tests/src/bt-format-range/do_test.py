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

    os.mkdir("out_migrated")
    os.mkdir("out_all")
    os.mkdir("out_none")
    call_subprocess(test_config.CT_TOOL + " --format-range=migrated --out-root=./out_migrated vector_add.cu --cuda-include-path=" + test_config.include_path, single_case_text)
    call_subprocess(test_config.CT_TOOL + " --format-range=all --out-root=./out_all vector_add.cu  --cuda-include-path=" + test_config.include_path, single_case_text)
    call_subprocess(test_config.CT_TOOL + " -format-range=none --out-root=./out_none vector_add.cu --cuda-include-path=" + test_config.include_path, single_case_text)

    ret = is_sub_string(";\/\/ variable declearation", "./out_llvm/vector_add.dp.cpp")
    ret = is_sub_string(";\/\/ allocate device memory", "./out_google/vector_add.dp.cpp") or ret
    ret = is_sub_string(";// variable declearation", "./out_custom/vector_add.dp.cpp") or ret
    ret = is_sub_string("; // allocate device memory", "./out_custom/vector_add.dp.cpp") or ret
    ret = is_sub_string("; // variable declearation", "./out_custom/vector_add.dp.cpp") or ret
    ret = is_sub_string("; // allocate device memory", "./out_custom/vector_add.dp.cpp") or ret
    return not ret
def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
