# ====------------------ do_test.py ---------- *- Python -* ----------------===#
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


def test_api(api_name, reference):
    call_subprocess(test_config.CT_TOOL + " -query-api-mapping=" + api_name)
    if "CUDA API: " + api_name + "\n" + reference + "\n" != test_config.command_output:
        print("API query message check failed: ", api_name)
        print(test_config.command_output)
        return False
    print("API query message check passed: ", api_name)
    return True


def migrate_test():
    test_cases = [
        [
            "cudaStreamGetFlags",
            "Is migrated to: an expression statement which set the output parameter 'flags' to 0",
        ],
        [
            "cudaEventDestroy",
            "Is migrated to: dpct::destroy_event(event_ptr event)",
        ],
        [
            "__hfma",
            "Is migrated to: sycl::fma(genfloat a, genfloat b, genfloat c)",
        ],
        [
            "__hfma_sat",
            "Is migrated to: sycl::ext::intel::math::hfma_sat(sycl::half x, sycl::half y, sycl::half z)\n"
            + "There are multi kinds of migrations for this API with different migration options,\n"
            + "suggest to use the tool to migrate a API usage code to see more detail of the migration.",
        ],
    ]

    res = True
    for test_case in test_cases:
        res = res and test_api(test_case[0], test_case[1])
    return res


def build_test():
    return True


def run_test():
    return True
