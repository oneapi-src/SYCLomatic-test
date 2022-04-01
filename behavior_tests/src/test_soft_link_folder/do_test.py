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

def setup_test():
    change_dir(test_config.current_test)
    prepare_soft_link_folder()
    return True


def prepare_soft_link_folder():
    os.symlink("cuda_", "cuda")

def migrate_test():

    src = [os.path.join("cuda", "test_soft_link_folder.cu")]
    in_root = ""
    extra_args = ""
    return do_migrate(src, in_root, test_config.out_root, extra_args)

def build_test():
    return True

def run_test():
    return True