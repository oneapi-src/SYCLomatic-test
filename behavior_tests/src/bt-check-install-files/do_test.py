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

def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    return True

def migrate_test(single_case_text):
    install_root = os.path.join(os.path.dirname(shutil.which("dpct")), '..')

    res = True
    if platform.system() == 'Linux':
        if not os.path.isfile(os.path.join(install_root, 'env/bash-autocomplete.sh')):
            print('Cannot found file:' + os.path.join(install_root, 'env/bash-autocomplete.sh'))
            res = False
    if not os.path.isfile(os.path.join(install_root, 'extensions/opt_rules/forceinline.yaml')):
        print('Cannot found file:' + os.path.join(install_root, 'extensions/opt_rules/forceinline.yaml'))
        res = False
    if not os.path.isfile(os.path.join(install_root, 'extensions/opt_rules/intel_specific_math.yaml')):
        print('Cannot found file:' + os.path.join(install_root, 'extensions/opt_rules/intel_specific_math.yaml'))
        res = False

    return res

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True

