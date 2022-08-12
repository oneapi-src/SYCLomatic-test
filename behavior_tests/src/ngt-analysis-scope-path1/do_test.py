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
    # migrate with implicit --analysis-scope-path which defaults to --in-root
    # clean previous migration output
    if (os.path.exists(os.path.join("out"))):
        shutil.rmtree(os.path.join("out"))
    migrate_cmd = test_config.CT_TOOL + " --cuda-include-path=" + test_config.include_path + " " + os.path.join(
        "cuda", "call_device_func_outside.cu") + " --in-root=" + os.path.join(
            "cuda") + " --out-root=" + os.path.join("out")
    for analysis_scope in (os.path.join(".", "cuda"), "./non_exist_dir/abc"):
        ret = call_subprocess(
            test_config.CT_TOOL + " --cuda-include-path=" +
            test_config.include_path + " " +
            os.path.join("cuda", "call_device_func_outside.cu") +
            " --in-root=" + os.path.join(".") + " --analysis-scope-path=" +
            analysis_scope + " --out-root=" + os.path.join("out"))
        if ret or (not is_sub_string(
                "Error: The path for --analysis-scope-path is not the same as or a parent directory of --in-root",
                test_config.command_output)):
            return False
    return True


def build_test():
    return True


def run_test():
    return True