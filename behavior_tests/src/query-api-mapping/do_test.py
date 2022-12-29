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


def test_api(api_name, reference):
    call_subprocess(test_config.CT_TOOL + " -query-api-mapping=" + api_name)
    if reference != test_config.command_output:
        print("API query message check failed: " + api_name)
        return False
    return True


def migrate_test():
    res = True
    res = res and test_api("cudaEventSynchronize",
                           "CUDA API\n"
                           "========\n"
                           "    cudaError_t cudaEventSynchronize(...)\n"
                           "SYCL API\n"
                           "========\n"
                           "    void sycl::event::wait_and_throw(...)\n"
                           "example\n"
                           "========\n"
                           "    CUDA code:\n"
                           "        int main() {\n"
                           "          cudaEvent_t e;\n"
                           "          cudaEventCreate(&e);\n"
                           "          cudaEventSynchronize(e);\n"
                           "        }\n"
                           "    SYCL code:\n"
                           "        #include <sycl/sycl.hpp>\n"
                           "        #include <dpct/dpct.hpp>\n"
                           "        int main() {\n"
                           "          dpct::event_ptr e;\n"
                           "          e = new sycl::event();\n"
                           "          e->wait_and_throw();\n"
                           "        }")
    res = res and test_api("cudaStreamWaitEvent",
                           "CUDA API\n"
                           "========\n"
                           "    cudaError_t cudaStreamWaitEvent(...)\n"
                           "SYCL API\n"
                           "========\n"
                           "    event sycl::queue::ext_oneapi_submit_barrier(...)\n"
                           "example\n"
                           "========\n"
                           "    CUDA code:\n"
                           "        int main() {\n"
                           "          cudaStream_t s;\n"
                           "          cudaStreamCreate(&s);\n"
                           "          cudaEvent_t e;\n"
                           "          cudaEventCreate(&e);\n"
                           "          cudaStreamWaitEvent(s, e, 0);\n"
                           "        }\n"
                           "    SYCL code:\n"
                           "        #include <sycl/sycl.hpp>\n"
                           "        #include <dpct/dpct.hpp>\n"
                           "        int main() {\n"
                           "          dpct::queue_ptr s;\n"
                           "          s = dpct::get_current_device().create_queue();\n"
                           "          dpct::event_ptr e;\n"
                           "          e = new sycl::event();\n"
                           "          s->ext_oneapi_submit_barrier({*e});\n"
                           "        }")
    res = res and test_api("cudaFreeArray",
                           "CUDA API\n"
                           "========\n"
                           "    cudaError_t cudaFreeArray(...)\n"
                           "SYCL API\n"
                           "========\n"
                           "    delete\n"
                           "example\n"
                           "========\n"
                           "    CUDA code:\n"
                           "        int main() {\n"
                           "          cudaArray_t array;\n"
                           "          cudaChannelFormatDesc channel;\n"
                           "          cudaMallocArray(&array, &channel, 1, 1);\n"
                           "          cudaFreeArray(array);\n"
                           "        }\n"
                           "    SYCL code:\n"
                           "        #include <sycl/sycl.hpp>\n"
                           "        #include <dpct/dpct.hpp>\n"
                           "        int main() {\n"
                           "          dpct::image_matrix_p array;\n"
                           "          dpct::image_channel channel;\n"
                           "          array = new dpct::image_matrix(channel, sycl::range<2>(1, 1));\n"
                           "          delete array;\n"
                           "        }")
    return res


def build_test():
    return True


def run_test():
    return True
