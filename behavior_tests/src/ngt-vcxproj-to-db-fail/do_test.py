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

def migrate_test(single_case_text):
   
    call_subprocess("icacls ./source /deny Everyone\:\(WD\)")
    
    vcxproj_path = os.path.join(os.getcwd(), "source", "aaa.vcxproj")
    source_path = os.path.join(os.getcwd(), "source", "simple_foo.cu")
    call_subprocess(test_config.CT_TOOL + " " + source_path   + " --vcxprojfile=\"" + vcxproj_path  + "\" --cuda-include-path=" + test_config.include_path)
    call_subprocess("icacls ./source /grant Everyone\:\(WD\)")
    return is_sub_string("Cannot create CompilationDatabase", single_case_text.command_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True