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
from hashlib import md5

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    install_root = os.path.join(os.path.dirname(shutil.which("dpct")), '..')

    res = True
    res = res and os.path.isfile(os.path.join(install_root, 'env/bash-autocomplete.sh'))
    res = res and os.path.isfile(os.path.join(install_root, 'extensions/opt_rules/forceinline.yaml'))
    res = res and os.path.isfile(os.path.join(install_root, 'extensions/opt_rules/intel_specific_math.yaml'))

    return res

def build_test():
    return True

def run_test():
    return True

