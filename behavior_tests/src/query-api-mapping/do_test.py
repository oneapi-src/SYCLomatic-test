# ===------------------- do_test.py ---------- *- Python -* ----------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===-----------------------------------------------------------------------===#

from test_utils import *


def setup_test():
    change_dir(test_config.current_test)
    return True


def test_api(api_name, source_code, options, migrated_code):
    call_subprocess(
        test_config.CT_TOOL
        + " --cuda-include-path="
        + test_config.include_path
        + " --query-api-mapping="
        + api_name
    )
    expect = "CUDA API:\n"
    for code in source_code:
        expect += code + "\n"
    expect += "Is migrated to"
    if options.__len__() > 0:
        expect += " (with the option"
        for option in options:
            expect += " " + option
        expect += ")"
    expect += ":\n"
    for code in migrated_code:
        expect += code + "\n"
    if expect != test_config.command_output:
        print("API query message check failed: ", api_name)
        print("output:\n", test_config.command_output, "===end===\n")
        print("expect:\n", expect, "===end===\n")
        return False
    print("API query message check passed: ", api_name)
    return True


def migrate_test():
    test_cases = [
        [
            "cudaEventDestroy",
            [
                "  cudaEventDestroy(e /*cudaEvent_t*/);",
            ],
            [],
            ["  dpct::destroy_event(e);"],
        ],
        [
            "cudaStreamGetFlags",
            [
                "  cudaStreamGetFlags(s /*cudaStream_t*/, f /*unsigned int **/);",
            ],
            [],
            ["  *(f) = 0;"],
        ],
    ]

    res = True
    for test_case in test_cases:
        res = res and test_api(test_case[0], test_case[1], test_case[2], test_case[3])
    return res


def build_test():
    return True


def run_test():
    return True
