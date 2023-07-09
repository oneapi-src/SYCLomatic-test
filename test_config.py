# ====------ test_config.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#

#Only high level config, global variable and static variable are reserved in test_config
# The arguments
DPCXX_COM = ""             # Default compiler will set by set_default_compiler API call.
CT_TOOL = "dpct"         # The migration tool binary name
PYTHON_COM = "python3 "
suite_list_file = "test_suite_list.xml"   # The configuration file lists all the suite to run and corresponding options.
VERBOSE_LEVEL = 0       # Debug verbose level： 0: silent all the debug information. None 0: turn on all the debug information.
include_path = ""  # Specify the CUDA header file path.
cuda_ver = 0        # CUDA header file version.
test_option = ""    # Ref the option_mapping.json file table.
migrate_option = ""
option_map = ""       # Option mapping table. Ref: option_mapping.json
root_path = ""      # The root path of test repo.
timeout = 1200       # The time limit for each test case.

# The default device for the test. Device can be "opencl:cpu", "opencl:gpu" and "level_zero:gpu".
# For all the devices: https://intel.github.io/llvm-docs/EnvironmentVariables.html#sycl_device_filter
device_filter = "level_zero:gpu"


gpu_device = ["Gen9", "Gen12"]

# The gpu support double kernel type.
support_double_gpu = ["Gen9"]
