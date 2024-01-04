# ====------ do_test.py---------- *- Python -* ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#
import os
import tempfile

from test_utils import *
from test_config import CT_TOOL


def setup_test():
    change_dir(test_config.current_test)
    return True


def migrate_test():
    """
    Runs dpct with options for --sycl-file-extension and verfies the exisitance
    of migrated files with correct extension
    """
    sub_command = f"{test_config.CT_TOOL} main.cpp --out-root {{0}}"

    # commands to test three options to --sycl-file-extension and the defualt
    # behaviour
    cmds_n_results = {
        f"{sub_command} --sycl-file-extension=sycl-cpp": "main.sycl.cpp",
        f"{sub_command} --sycl-file-extension=dp-cpp": "main.dp.cpp",
        f"{sub_command} --sycl-file-extension=cpp": "main.cpp",
        sub_command: "main.dp.cpp",
    }

    # run command and verify the existance of migarate file with expected
    # extension
    for cmd, expected_file in cmds_n_results.items():
        with tempfile.TemporaryDirectory() as dpct_out_root:
            cmd = cmd.format(dpct_out_root)
            ret = call_subprocess(cmd)

            if not ret:
                print(f"user-defined-sycl-file-extension: cmd execution failed: {cmd}")
                return False

            if not os.path.exist(os.path.join(dpct_out_root, expected_file)):
                print(f"user-defined-sycl-file-extension: test failed: {cmd}")
                return False

    # check for incorrect value to --sycl-file-extension
    with tempfile.TemporaryDirectory() as dpct_out_root:
        cmd = f"{subcommand} --sycl-file-extension=incorrect-option"
        call_subprocess(cmd)
        return is_sub_string(
            "Cannot find option named 'incorrect-option'", test_config.command_output
        )

    return True


def build_test():
    return True


def run_test():
    return True
