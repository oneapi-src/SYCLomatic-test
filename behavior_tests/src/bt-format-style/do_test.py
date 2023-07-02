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
    os.mkdir("out_custom")
    os.mkdir("out_llvm")
    os.mkdir("out_google")
    call_subprocess(test_config.CT_TOOL + " --format-style=custom --out-root=./out_custom vector_add.cu  --cuda-include-path=" + test_config.include_path)
    call_subprocess(test_config.CT_TOOL + " --format-style=llvm --out-root=./out_llvm vector_add.cu  --cuda-include-path=" + test_config.include_path)
    call_subprocess(test_config.CT_TOOL + " --format-style=google --out-root=./out_google vector_add.cu  --cuda-include-path=" + test_config.include_path)
    ret = False
    with open(os.path.join("out_llvm", "vector_add.dp.cpp")) as f:
        file_str = f.read()
        ret = is_sub_string("; //", file_str)

    with open(os.path.join("out_google", "vector_add.dp.cpp")) as f:
        file_str = f.read()
        ret = is_sub_string("; //", file_str) or ret

    with open(os.path.join("out_custom", "vector_add.dp.cpp")) as f:
        file_str = f.read()
        ret = is_sub_string("; //", file_str) or ret

    return ret
def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
