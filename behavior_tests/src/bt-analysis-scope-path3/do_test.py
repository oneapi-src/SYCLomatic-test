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
    return True


def migrate_test(single_case_text):
    # clean previous migration output
    if (os.path.exists("out")):
        shutil.rmtree("out")
    migrate_cmd = test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " " + os.path.join(
        "cuda",
        "call_device_func_outside.cu") + " --in-root=cuda" + " --out-root=out"

    call_subprocess(migrate_cmd + " --analysis-scope-path=" +
                    os.path.join("cuda", ".."))
    expected_files = [
        os.path.join("out", "call_device_func_outside.dp.cpp"),
    ]
    collected_files = []
    for ext in ('.cpp', '.cu', '.h', 'hpp'):
        collected_files += glob.glob("out/**/*" + ext, recursive=True)
    if not collected_files:
        print("Cannot collect migrated files in --out-root " +
              os.path.realpath("out"))
        return False

    # Check if unexpected files were generated or copied from --in-root
    error_msg = ""
    for file in collected_files:
        if file not in expected_files:
            error_msg = error_msg + "\n" + file + " is not expected to be in --out-root " + os.path.realpath(
                "out")
    if len(error_msg):
        print(error_msg)
        return False
    return True


def build_test(single_case_text):
    return True


def run_test(single_case_text):
    return True