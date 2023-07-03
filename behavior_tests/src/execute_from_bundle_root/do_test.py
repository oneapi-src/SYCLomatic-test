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

def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    prepare_execution_folder()
    return True


def prepare_execution_folder():
    copy_ct_bin = os.path.dirname(str(shutil.which(test_config.CT_TOOL)))
    distutils.dir_util.copy_tree(copy_ct_bin, "bin")

def migrate_test(single_case_text):
    ct_bin = os.path.join("bin", test_config.CT_TOOL)
    in_root = ""
    extra_args = ""
    call_subprocess(ct_bin + " --cuda-include-path=" + test_config.include_path +
                " " + "hellocuda.cu", single_case_text)
    if ('or the same folder as' in single_case_text.print_text):
        return True
    return False

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True