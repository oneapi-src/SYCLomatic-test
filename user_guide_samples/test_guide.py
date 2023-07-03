# ====------ test_guide.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import os
import re
import sys
from pathlib import Path
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

def setup_test(single_case_text):
    return True

def migrate_test(single_case_text):
    return True

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    test_driver = single_case_text.name + ".py"
    options = " \" \""
    os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
    call_subprocess("python " + test_driver + options, single_case_text)
    if "case pass" in single_case_text.print_text:
        return True
    return False
