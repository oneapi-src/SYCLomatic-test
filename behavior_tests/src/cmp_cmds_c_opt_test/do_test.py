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
import json

from test_utils import *

def setup_test():
    change_dir(test_config.current_test)
    return True

def parse_compilation_database(compilation_database_name):
  if (not os.path.isfile(compilation_database_name)):
    print("The compilation database is not existed, please double check\n")

  with open(compilation_database_name) as compilation_database:
    compilation_database_items = json.load(compilation_database)
    database_source_files = []
    for entry in compilation_database_items:
      compilation_item = entry['file'];
      database_source_files.append(os.path.basename(compilation_item))
    return database_source_files

def migrate_test():
    call_subprocess("intercept-build make -f Makefile CC=nvcc USE_SM=70 -B")
    new_database = parse_compilation_database("compile_commands.json")
    reference_database = parse_compilation_database("compile_commands.json_ref")
    new_database = list(set(new_database))
    reference_database = list(set(reference_database))
    new_database.sort()
    reference_database.sort()
    if(new_database == reference_database):
        return True
    return False

def build_test():
    return True

def run_test():
    return True
