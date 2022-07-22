# ====------ test_help.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import os
import re
import sys
from pathlib import Path
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

def setup_test():
    return True

def migrate_test():
    return True

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    test_config.out_root = os.getcwd()

    lpthread_link_cases = ["devicemgr_multi_thread_aware", "memory_async_dpct_free", "blas_extension_api_usm",
                           "blas_extension_api_buffer"]
    oneDPL_related = ["thrust_test_device_ptr_2", "thrust_test-pennet_simple_pstl", "test_default_queue_2"]
    blas_cases = ["blas_utils_getrf", "blas_utils_getrf-usm", "blas_utils_getrf-complex",
                "blas_utils_getrf-complex-usm", "blas_utils_getrs", "blas_utils_getrs-usm",
                "blas_utils_getrs-complex", "blas_utils_getrs-complex-usm", "blas_utils_getri",
                "blas_utils_getri-usm", "blas_utils_getri-complex", "blas_utils_getri-complex-usm",
                "blas_utils_geqrf", "blas_utils_geqrf-usm", "blas_utils_geqrf-complex",
                "blas_utils_geqrf-complex-usm", "blas_utils_get_transpose", "blas_utils_get_value",
                "blas_utils_get_value_usm", "blas_extension_api_buffer", "lib_common_utils_mkl_get_version",
                "blas_extension_api_usm", "blas_utils_getrfnp", "blas_utils_getrfnp-usm", "blas_utils_getrfnp-complex",
                "blas_utils_getrfnp-complex-usm"]
    oneDNN_related = ["dnnl_utils_activation", "dnnl_utils_fill", "dnnl_utils_lrn", "dnnl_utils_memory",
                "dnnl_utils_pooling", "dnnl_utils_reorder", "dnnl_utils_scale", "dnnl_utils_softmax",
                "dnnl_utils_sum"]

    srcs = []
    cmp_opts = []
    link_opts = []
    objects = []

    if test_config.current_test in oneDPL_related:
        cmp_opts.append(prepare_oneDPL_specific_macro())
    if test_config.current_test in lpthread_link_cases and platform.system() == "Linux":
        link_opts.append("-lpthread")

    for dirpath, dirnames, filenames in os.walk(test_config.out_root):
        srcs.append(os.path.abspath(os.path.join(dirpath, test_config.current_test + ".cpp")))
    ret = False

    if test_config.current_test == "test_default_queue_2":
        srcs.append("test_default_queue_1.cpp")
    if test_config.current_test == "kernel_function_lin":
        ret = call_subprocess("dpcpp -shared -fPIC -o module.so kernel_module_lin.cpp")
        if not ret:
            print("kernel_function_lin created the shared lib failed.")
            return False
    if test_config.current_test == "kernel_function_win":
        ret = call_subprocess("dpcpp /EHsc -shared -o module.dll kernel_module_win.cpp")
        if not ret:
            print("kernel_function_win created the shared lib failed.")
            return False
    if test_config.current_test in oneDNN_related:
        link_opts.append(" -ldnnl")
    if test_config.current_test in blas_cases:
        mkl_opts = []
        if platform.system() == "Linux":
            mkl_opts = test_config.mkl_link_opt_lin
        else:
            mkl_opts = test_config.mkl_link_opt_win

        link_opts += mkl_opts
        cmp_opts.append("-DMKL_ILP64")
    ret = compile_and_link(srcs, cmp_opts, objects, link_opts)
    return ret


def run_test():
    os.environ["SYCL_DEVICE_FILTER"] = test_config.device_filter
    args = []
    if test_config.current_test == "kernel_function_lin":
        args.append("./module.so")
    if test_config.current_test == "kernel_function_win":
        args.append("./module.dll")

    ret = run_binary_with_args(args)
    if test_config.current_test == "async_exception" and "Caught asynchronous SYCL exception" in test_config.command_output and "test_dpct_async_handler" in test_config.command_output:
        return True
    return ret

