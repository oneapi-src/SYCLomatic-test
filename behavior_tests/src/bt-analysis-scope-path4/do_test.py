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
import shutil
import glob

from test_utils import *


def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    if os.path.exists("cuda_symlink"):
        os.unlink("cuda_symlink")
    os.symlink("cuda", "cuda_symlink")
    return True


def migrate_test(single_case_text):
    # clean previous migration output
    if (os.path.exists("out")):
        shutil.rmtree("out")

    migrate_cmd = single_case_text.CT_TOOL + " --cuda-include-path=" + single_case_text.include_path + " " + os.path.join(
        "cuda",
        "call_device_func_outside.cu") + " --in-root=cuda" + " --out-root=out"

    call_subprocess(migrate_cmd + " --analysis-scope-path=cuda_symlink", single_case_text)
    if (not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False
    return True


def build_test(single_case_text):
    return True


def run_test(single_case_text):
    return True