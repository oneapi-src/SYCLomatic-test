# ====------ foo.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import subprocess
import os
import sys
from os import path
import platform

p = "--cuda-include-path=\"" + os.environ["CUDA_INCLUDE_PATH"] + "\" " + sys.argv[1]

# migrate code
in_root = os.path.join(os.getcwd(), "foo")
run_shell=True
if platform.system() == "Windows":
    run_shell=False
default_out_root = "result"
main_src = os.path.join("foo", "main.cu")
util_src = os.path.join("foo", "bar", "util.cu")
header_path = "result"
cmd = "dpct --in-root=\"" + in_root + "\" --out-root=\"" + default_out_root +    \
                    "\" \"" + main_src + "\" \"" + util_src + "\" " + p + " -- -I \"" + header_path + "\""
print(cmd)
complete_process = subprocess.run(cmd, shell=run_shell, check=False,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

print(complete_process.stdout)
print("Migration done!!")

main_file = os.path.join("result", "main.dp.cpp")
main_yml_file = os.path.join("result", "MainSourceFiles.yaml")
util_cpp_file = os.path.join("result", "bar", "util.dp.cpp")
util_h_file = os.path.join("result", "bar", "util.h")
ret = path.exists(main_file) and \
      path.exists(main_yml_file) and \
      path.exists(util_cpp_file) and \
      path.exists(util_h_file)

# check run result
if ret:
    print("foo case pass")
else:
    print("foo case fail")

