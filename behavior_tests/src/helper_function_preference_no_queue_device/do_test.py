# ===---------- do_test.py ---------- *- Python -* -------------------------===#
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


def check_file_contain_or_not(file, str, flag):
    with open(file, "r") as f:
        content = f.read()
        if flag:
            if str not in content:
                print("there should be '" + str + "' in file " + file)
                return False
        else:
            if str in content:
                print("there should not be '" + str + "' in file " + file)
                return False
    return True


def migrate_test():
    call_subprocess(
        test_config.CT_TOOL
        + " -p=. --out-root=out_default --cuda-include-path="
        + test_config.include_path
    )

    call_subprocess(
        test_config.CT_TOOL
        + " --helper-function-preference=no-queue-device -p=. --out-root=out_no_q_d --cuda-include-path="
        + test_config.include_path
    )

    res = True

    # Check "dpct::get_current_device" without option.
    res = res and check_file_contain_or_not(
        "out_default/main.dp.cpp", "dpct::get_current_device", True
    )
    res = res and check_file_contain_or_not(
        "out_default/kernel1.dp.cpp", "dpct::get_current_device", True
    )
    res = res and check_file_contain_or_not(
        "out_default/kernel2.dp.cpp", "dpct::get_current_device", True
    )

    # Check "dpct::get_current_device" with option --helper-function-preference=no-queue-device.
    res = res and check_file_contain_or_not(
        "out_no_q_d/main.dp.cpp", "dpct::get_current_device", False
    )
    res = res and check_file_contain_or_not(
        "out_no_q_d/kernel1.dp.cpp", "dpct::get_current_device", False
    )
    res = res and check_file_contain_or_not(
        "out_no_q_d/kernel2.dp.cpp", "dpct::get_current_device", False
    )

    # Check "dpct::get_in_order_queue" without option.
    res = res and check_file_contain_or_not(
        "out_default/kernel1.dp.cpp", "dpct::get_in_order_queue", True
    )
    res = res and check_file_contain_or_not(
        "out_default/kernel2.dp.cpp", "dpct::get_in_order_queue", True
    )

    # Check "dpct::get_in_order_queue" with option --helper-function-preference=no-queue-device.
    res = res and check_file_contain_or_not(
        "out_no_q_d/main.dp.cpp", "dpct::get_in_order_queue", False
    )
    res = res and check_file_contain_or_not(
        "out_no_q_d/kernel1.dp.cpp", "dpct::get_in_order_queue", False
    )
    res = res and check_file_contain_or_not(
        "out_no_q_d/kernel2.dp.cpp", "dpct::get_in_order_queue", False
    )

    # Check "dpct::device_ext" without option.
    res = res and check_file_contain_or_not(
        "out_default/main.dp.cpp", "dpct::device_ext", True
    )
    res = res and check_file_contain_or_not(
        "out_default/kernel1.dp.cpp", "dpct::device_ext", True
    )
    res = res and check_file_contain_or_not(
        "out_default/kernel2.dp.cpp", "dpct::device_ext", True
    )

    # Check "dpct::device_ext" with option --helper-function-preference=no-queue-device.
    res = res and check_file_contain_or_not(
        "out_no_q_d/main.dp.cpp", "dpct::device_ext", False
    )
    res = res and check_file_contain_or_not(
        "out_no_q_d/kernel1.dp.cpp", "dpct::device_ext", False
    )
    res = res and check_file_contain_or_not(
        "out_no_q_d/kernel2.dp.cpp", "dpct::device_ext", False
    )

    return res


def build_test():
    return True


def run_test():
    return True
