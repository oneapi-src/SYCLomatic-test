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
    call_subprocess(test_config.CT_TOOL + " hello.h --out-root=out --cuda-include-path=" + \
                   os.environ["CUDA_INCLUDE_PATH"])
    migrated_yaml = os.path.join("out", "hello.h.yaml")
    with open(migrated_yaml, "r") as f:
        file_data = f.read()

    file_data = file_data.replace("hello.h", "hello_aaa.h")
    with open(migrated_yaml, "w") as f:
        f.write(file_data)

    call_subprocess(test_config.CT_TOOL + " hello.h --out-root=out --cuda-include-path=" + \
                   os.environ["CUDA_INCLUDE_PATH"])
    if "differnt path" in single_case_text.command_text:
        return True
    print("not catch the error: unkown key constantFla")
    return False


def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
