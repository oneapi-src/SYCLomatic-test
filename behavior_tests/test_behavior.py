# ====------ test_behavior.py---------- *- Python -* ----===##
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
    src = []
    extra_args = []
    in_root = os.path.join(os.getcwd(), single_case_text.name)
    single_case_text.out_root = os.path.join(in_root, 'out_root')


    for dirpath, dirnames, filenames in os.walk(in_root):
        for filename in [f for f in filenames if re.match('.*(cu|cpp|c)$', f)]:
            src.append(os.path.abspath(os.path.join(dirpath, filename)))
    return do_migrate(src, in_root, single_case_text.out_root, single_case_text, extra_args)

def build_test(single_case_text):
    if (os.path.exists(single_case_text.name)):
        os.chdir(single_case_text.name)
    srcs = []
    cmp_opts = ''
    link_opts = ''
    objects = ''

    for dirpath, dirnames, filenames in os.walk(single_case_text.out_root):
        for filename in [f for f in filenames if re.match('.*(cpp|c)$', f)]:
            srcs.append(os.path.abspath(os.path.join(dirpath, filename)))
    ret = False
    ret = compile_files(srcs, single_case_text, cmp_opts)
    return ret

def run_test(single_case_text):
    return True
