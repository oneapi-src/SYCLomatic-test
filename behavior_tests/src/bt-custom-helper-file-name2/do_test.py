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
    path_env =""
    split_val = ""
    call_subprocess(test_config.CT_TOOL + " test.cu --use-custom-helper=file --custom-helper-name=my_proj --out-root=out --cuda-include-path=" + test_config.include_path)
    if (platform.system() == 'Linux'):
        path_env = str(os.environ["CPATH"])
        split_val = ":"
    else:
        path_env = str(os.environ["INCLUDE"])
        split_val = ";"
    env_list = path_env.split(split_val)
    ret = []
    for env in env_list:
        if not (("deploy_" in env) or ("dpcpp-ct" in env)):
            ret.append(env)
    ret.append(os.path.join(os.getcwd(), "out", "include"))
    header_path = split_val.join(ret)
    ret_val = False
    if (platform.system() == 'Windows'):
        os.environ["INCLUDE"] = header_path
        print(os.environ["INCLUDE"])
        ret_val = call_subprocess("icx-cl -fsycl /EHsc out/test.dp.cpp -o out/run")
        os.environ["INCLUDE"] = path_env
    else:
        os.environ["CPATH"] = header_path
        call_subprocess("env | grep CPATH")
        ret_val = call_subprocess("icpx -fsycl out/test.dp.cpp -o out/run")
        os.environ["CPATH"] = path_env
    return ret_val

def build_test():
    return True

def run_test():
    return True