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

from test_utils import *


def setup_test():
    change_dir(test_config.current_test)
    return True


def migrate_test():
    # clean previous migration output
    if (os.path.exists(os.path.join("out"))):
        shutil.rmtree(os.path.join("out"))
    migrate_cmd = test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " " + os.path.join(
        "cuda", "call_device_func_outside.cu") + " --in-root=" + os.path.join(
            "cuda") + " --out-root=" + os.path.join("out")
    # migrate with implicit --analysis-scope-path which defaults to --in-root
    call_subprocess(migrate_cmd)
    if (not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False

    # expect incremental migration with specified --analysis-scope-path which equals --in-root
    ret = call_subprocess(migrate_cmd + " --analysis-scope-path=" +
                          os.path.join("cuda"))
    if (not ret or not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False

    # not expect incremental migration with specified --analysis-scope-path which is the parent of --in-root
    ret = call_subprocess(migrate_cmd + " --analysis-scope-path=" +
                          os.path.join("cuda", ".."))
    previous_analysis_scope = os.path.realpath(os.path.join("cuda"))
    print(test_config.command_output)
    if ret or not is_sub_string(
            f"use the same option set as in previous migration: \"--analysis-scope-path=\"{previous_analysis_scope}\"",
            test_config.command_output):
        return False
    return True


def build_test():
    return True


def run_test():
    return True