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
    test_dir = os.path.join(os.getcwd())
    ref_cmp_db_file = open(test_dir + "/compile_commands.json_ref", "rt")
    cmp_cmds = ref_cmp_db_file.read()
    cmp_cmds = cmp_cmds.replace("${TEST_DIRECTORY}", test_dir)
    ref_cmp_db_file.close()
    ref_cmp_db_file = open(test_dir + "/compile_commands.json_ref", "wt")
    ref_cmp_db_file.write(cmp_cmds)
    ref_cmp_db_file.close()

    call_subprocess(test_config.CT_TOOL + " --intercept-build make")
    result1 = os.path.isfile(
        os.path.join(test_dir, "compile_commands.json")
    ) and filecmp.cmp(
        test_dir + "/compile_commands.json",
        test_dir + "/compile_commands.json_ref",
        shallow=False,
    )
    if result1 is not True:
        print(
            "intercept-build not successful for command: ",
            test_config.CT_TOOL + " --intercept-build make",
        )
        return False
    
    call_subprocess("make clean")
    call_subprocess(test_config.CT_TOOL + " -intercept-build -vv make -B")
    result2 = (
        is_sub_string("verbose=2", test_config.command_output)
        and os.path.isfile(os.path.join(test_dir, "compile_commands.json"))
        and filecmp.cmp(
            test_dir + "/compile_commands.json",
            test_dir + "/compile_commands.json_ref",
            shallow=False,
        )
    )
    if result2 is not True:
        print(
            "intercept-build not successful for command: ",
            test_config.CT_TOOL + " -intercept-build -vv make -B",
        )
        return False

    call_subprocess("make clean")
    call_subprocess(
        test_config.CT_TOOL
        + " intercept-build -vvv --cdb compile_commands2.json make -B"
    )
    result3 = (
        is_sub_string("verbose=3", test_config.command_output)
        and is_sub_string(
            """cdb='compile_commands2.json'""",
            test_config.command_output,
        )
        and os.path.isfile(os.path.join(test_dir, "compile_commands2.json"))
        and filecmp.cmp(
            test_dir + "/compile_commands2.json",
            test_dir + "/compile_commands.json_ref",
            shallow=False,
        )
    )
    if result3 is not True:
        print(
            "intercept-build not successful for command: ",
            test_config.CT_TOOL
            + " intercept-build -vvv --cdb compile_commands2.json make -B",
        )
        return False
    return True

def build_test():
    return True

def run_test():
    return True
