# ====------------------ do_test.py---------- *- Python -* ---------------===##
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
    change_dir(single_case_text.name, single_case_text)
    return True


def migrate_test(single_case_text):
    res = True
    cmd = test_config.CT_TOOL + " --out-root=out double.cu --cuda-include-path=" + test_config.include_path
    if not call_subprocess(cmd):
        res = False
        print("double.cu migrate failed")
    else:
        print("double.cu migrate pass")
    cmd = test_config.CT_TOOL + " --out-root=out half.cu --cuda-include-path=" + test_config.include_path
    if not call_subprocess(cmd):
        res = False
        print("half.cu migrate failed")
    else:
        print("half.cu migrate pass")
    return res


def build_test(single_case_text):
    res = True
    if not compile_files([os.path.join("out", "double.dp.cpp")], single_case_text):
        res = False
        print("double.dp.cpp compile failed")
    else:
        print("double.dp.cpp compile pass")
    cmd = test_config.DPCXX_COM + ' ' + prepare_obj_name(os.path.join("out", "double.dp.cpp")) + " -o double.run"
    if not call_subprocess(cmd):
        res = False
        print("double.dp.cpp link failed")
    else:
        print("double.dp.cpp link pass")
    if not compile_files([os.path.join("out", "half.dp.cpp")], single_case_text):
        res = False
        print("half.dp.cpp compile failed")
    else:
        print("half.dp.cpp compile pass")
    cmd = test_config.DPCXX_COM + ' ' + prepare_obj_name(os.path.join("out", "half.dp.cpp")) + " -o half.run"
    if not call_subprocess(cmd):
        res = False
        print("half.dp.cpp link failed")
    else:
        print("half.dp.cpp link pass")
    return res


def run_test(single_case_text):
    os.environ["ONEAPI_DEVICE_SELECTOR"] = test_config.device_filter
    res = 0

    cmd = os.path.join(os.path.curdir, 'double.run')
    if call_subprocess(cmd):
        res += 1
        print("double.run run pass")
    print("double.run output:")
    print(single_case_text.command_text)
    if res != 1:
        print("case 'double' failed")
        return False

    cmd = os.path.join(os.path.curdir, 'half.run')
    if call_subprocess(cmd):
        res += 1
        print("half.run run pass")
    print("half.run output:")
    print(single_case_text.command_text)
    if res != 2:
        print("case 'half' failed")
        return False

    return True
