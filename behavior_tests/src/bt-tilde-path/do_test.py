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
old_tilde = ""
def setup_test(single_case_text):
    global old_tilde
    change_dir(single_case_text.name, single_case_text)
    old_tilde = os.path.expanduser("~")
    os.environ["HOME"] = "/tmp"
    return True

def migrate_test(single_case_text):
    global user_root_path
    user_root_path = os.path.expanduser("~") + "/workspace"
    if os.path.exists(user_root_path):
        shutil.rmtree(user_root_path)
    os.makedirs(user_root_path)
    distutils.dir_util.copy_tree(test_config.include_path + "/..", user_root_path)
    return call_subprocess("dpct vector_add.cu --cuda-include-path=" + "~/workspace/include")

def build_test(single_case_text):
    global old_tilde
    global user_root_path
    if os.path.exists(user_root_path):
        shutil.rmtree(user_root_path)
    os.environ["HOME"] = old_tilde
    return True

def run_test(single_case_text):
    return True