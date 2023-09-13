# ====------ do_test.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import filecmp
import subprocess
import platform
import os
import sys

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def migrate_test():
    call_subprocess(test_config.CT_TOOL + " --intercept-build -vvv make")
    test_dir = os.path.join(os.getcwd())
    ref_cmp_db_file = open(test_dir+"/compile_commands.json_ref", "rt")
    cmp_cmds = ref_cmp_db_file.read()
    cmp_cmds = cmp_cmds.replace('${TEST_DIRECTORY}', test_dir)
    ref_cmp_db_file.close()
    ref_cmp_db_file = open(test_dir+"/compile_commands.json_ref", "wt")
    ref_cmp_db_file.write(cmp_cmds)
    ref_cmp_db_file.close()

    result = filecmp.cmp(test_dir+"/compile_commands.json",
                         test_dir+"/compile_commands.json_ref", shallow=False)
    
    return result

def build_test():
    return True

def run_test():
    return True
