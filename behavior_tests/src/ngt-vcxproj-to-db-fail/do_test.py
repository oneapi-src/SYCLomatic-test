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
    return True

def migrate_test():
   
    call_subprocess("icacls ./source /deny Everyone\:\(WD\)")
    
    vcxproj_path = os.path.join(os.getcwd(), "source", "aaa.vcxproj")
    source_path = os.path.join(os.getcwd(), "source", "simple_foo.cu")
    call_subprocess(test_config.CT_TOOL + " " + source_path   + " --vcxprojfile=\"" + vcxproj_path  + "\" --cuda-include-path=" + test_config.include_path)
    call_subprocess("icacls ./source /grant Everyone\:\(WD\)")
    return is_sub_string("Cannot create CompilationDatabase", test_config.command_output)

def build_test():
    return True

def run_test():
    return True