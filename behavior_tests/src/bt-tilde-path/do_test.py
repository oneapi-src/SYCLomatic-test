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
import glob

from test_utils import *
user_root_path = ""
def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    global user_root_path
    user_root_path = os.path.expanduser("~") + "/workspace"
    if os.path.exists(user_root_path):
        shutil.rmtree(user_root_path)
    os.makedirs(user_root_path)
    distutils.dir_util.copy_tree(test_config.include_path + "/..", user_root_path)
    return call_subprocess("dpct vector_add.cu --cuda-include-path=" + "~/workspace/include")

def build_test():
    global user_root_path
    if os.path.exists(user_root_path):
        shutil.rmtree(user_root_path)
    return True

def run_test():
    return True