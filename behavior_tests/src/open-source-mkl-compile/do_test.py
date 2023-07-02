# ====----------------- do_test.py---------- *- Python -* ----------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import os

from test_utils import *


def setup_test(single_case_text):
    print("ONEMKLROOT =", end = " ")
    print(os.environ["ONEMKLROOT"])
    if not os.path.exists(os.environ["ONEMKLROOT"] + "/include"):
        print("The path '${ONEMKLROOT}/include' is not exist!")
        return False
    change_dir(single_case_text.name, single_case_text)
    return True


def migrate_test(single_case_text):
    return True


def build_test(single_case_text):
    cmp_opts = ["-I${ONEMKLROOT}/include"]
    ret = False
    ret = compile_files(["main.dp.cpp"], single_case_text, cmp_opts)
    return ret


def run_test(single_case_text):
    return True
