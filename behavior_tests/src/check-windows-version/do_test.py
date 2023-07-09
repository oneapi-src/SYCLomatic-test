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

def get_windows_version(arg1, arg2, single_case_text):
    call_subprocess("powershell \"(Get-Item -path " + arg1 + ").VersionInfo." + arg2 + "\"", single_case_text)
    return single_case_text.print_text

def migrate_test(single_case_text):
    ct_path = get_ct_path()
    if not ct_path:
        print('Cannot find the path to dpct!')
        return False
    ct_clang_version = get_ct_clang_version()
    if not ct_path:
        print("Cannot find dpct's bundled clang version!")
        return False
    print("dpct's bundled clang version is: {}".format(ct_clang_version))

    file_version = get_windows_version(ct_path, 'FileVersion', single_case_text).strip()
    product_version = get_windows_version(ct_path, 'ProductVersion', single_case_text).strip()
    product_name = get_windows_version(ct_path, 'ProductName', single_case_text).strip()
    file_description = get_windows_version(ct_path, 'FileDescription', single_case_text).strip()
    legal_copyright = get_windows_version(ct_path, 'LegalCopyright', single_case_text).strip()

    print("====={}'s VersionInfo properties=====".format(ct_path))
    print("FileVersion: {}".format(file_version))
    print("ProductVersion: {}".format(product_version))
    print("ProductName: {}".format(product_name))
    print("FileDescription: {}".format(file_description))
    print("LegalCopyright: {}".format(legal_copyright))
    print("==========")

    return file_version == ct_clang_version and product_version == ct_clang_version and \
        product_name == "SYCLomatic" and file_description == "" and legal_copyright == ""

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True
