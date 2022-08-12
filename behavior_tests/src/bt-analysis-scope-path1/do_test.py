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
    shutil.rmtree(os.path.join("out"))

    # migrate with specified --analysis-scope-path which equals --in-root
    ret = call_subprocess(migrate_cmd + " --analysis-scope-path=" +
                          os.path.join("cuda"))
    if (not ret or not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False
    shutil.rmtree(os.path.join("out"))

    # migrate with specified --analysis-scope-path which is the parent of --in-root
    call_subprocess(migrate_cmd + " --analysis-scope-path=" +
                    os.path.join("cuda", ".."))
    if (not os.path.exists(
            os.path.join("out", "call_device_func_outside.dp.cpp"))):
        return False
    return True


def build_test():
    return True


def run_test():
    return True