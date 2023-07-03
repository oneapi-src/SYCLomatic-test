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

def setup_test(single_case_text):
    change_dir(single_case_text.name, single_case_text)
    return True

def migrate_test(single_case_text):
    call_subprocess(test_config.CT_TOOL + " test.cu --in-root . --out-root out --always-use-async-handler --assume-nd-range-dim=1 --comments --enable-ctad --no-dpcpp-extensions=enqueued_barriers --no-dry-pattern --process-all -p . --sycl-named-lambda --use-experimental-features=free-function-queries,nd_range_barrier --use-explicit-namespace=cl,dpct --usm-level=none --cuda-include-path=" + test_config.include_path, single_case_text)
    call_subprocess(test_config.CT_TOOL + " test.cu --out-root out --cuda-include-path=" + test_config.include_path, single_case_text)
    return is_sub_string("\"--analysis-scope-path=\"", single_case_text.command_text) and \
           is_sub_string("--always-use-async-handler --comments --compilation-database=\"", single_case_text.command_text) and \
           is_sub_string("--enable-ctad --use-experimental-features=free-function-queries,nd_range_barrier --use-explicit-namespace=cl,dpct --no-dpcpp-extensions=enqueued_barriers --assume-nd-range-dim=1 --no-dry-pattern --process-all --sycl-named-lambda --usm-level=none\".", single_case_text.command_text)

def build_test(single_case_text):
    return True

def run_test(single_case_text):
    return True