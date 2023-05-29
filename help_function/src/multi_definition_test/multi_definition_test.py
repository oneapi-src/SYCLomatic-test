# ====------ multi_definition_test.py------------------ *- Python -* ----===##
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

# prepare env
result = subprocess.call("/usr/bin/make")
print("Build done!!")

# # check run result
if result == 0:
    print("case pass")
else:
    print("case fail")
