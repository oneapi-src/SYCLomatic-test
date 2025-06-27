# ====------ test_feature.py---------- *- Python -* ----===##
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
import fileinput
import shutil

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *
cmpl_only_tests =['thrust-for-h2o4gpu', 'cublas-create-Sgemm-destroy', 'cublasLegacyLv123', 'cublasReturnType', 'cublasTtrmm', 'cublas_64',
                 'grid_constant', 'cusolver_range', 'cusparse-helper', 'macro', 'volatile-vec']

cmpllink_only_as_wa_tests = ['thrust-op', 'curand', 'curand-usm', 'curand-cross-function', 'cublasBatch', 'cublasGetSetMatrix', 'cublasGetSetVector', 'cublasIsamax_etc', 'cublas-lambda',
                        'cublas_curandInMacro', 'cublasLegacyCZ', 'cublasLegacyHelper', 'cublasRegularCZ', 'cublasTsyrkx', 'cublasTtrmmLegacy', 'cublas-usm-legacy', 'math-emu', 'cublas-usm',
                        'cusolverDnEi', 'cusolverDnEi-part2', 'cusolverDnLn', 'cusolverDnLn_cuda10-1', 'cusolverDnLn_cuda10-1-part2', 'cusolverDnLn-part2', 'cublas-only-usm', 'cufft-deduce',
                         'cufft-different-locations-usm', 'cufft-reuse-handle', 'cufft-different-locations', 'cufft-usm', 'cufft', 'cooperative_groups', 'driverCtx', 'driverDevice', 'driverArray', 'driverTex',
                         'driverMemset', 'cub_warp', 'cub_device_run_length_encode_encode', 'cooperative_group_coalesced_group', 'text_experimental_build_only',
                         'wmma', 'assert', 'peer_access_using_driver_api', 'curand-device-usm', 'curand-device', 'cufft-func-ptr', 'cufft-others', 'cub_block', 'codepin_basic', 'math-direct', 'math-helper', 'grid_sync_root_group', 'device_cpu', 'image']

occupancy_calculation_exper = ['occupancy_calculation']

def setup_test():
    if test_config.current_test == 'user_defined_rules_2':
        call_subprocess('mkdir -p user_defined_rules_2/ATen/cuda')
        call_subprocess('mkdir -p user_defined_rules_2/c10/core')
        call_subprocess('mkdir -p user_defined_rules_2/c10/cuda')
        call_subprocess('mkdir -p user_defined_rules_2/c10/util')
        call_subprocess('mkdir -p user_defined_rules_2/c10/xpu')
        call_subprocess('mkdir -p user_defined_rules_2/src')
        call_subprocess('mv user_defined_rules_2/CUDATensorMethods.cuh user_defined_rules_2/ATen/cuda')
        call_subprocess('mv user_defined_rules_2/Tensor.h user_defined_rules_2/ATen')
        call_subprocess('mv user_defined_rules_2/DeviceGuard.h user_defined_rules_2/c10/core')
        call_subprocess('mv user_defined_rules_2/Device.h user_defined_rules_2/c10/core')
        call_subprocess('mv user_defined_rules_2/CUDAGuard.h user_defined_rules_2/c10/cuda')
        call_subprocess('mv user_defined_rules_2/CUDAStream.h user_defined_rules_2/c10/cuda')
        call_subprocess('mv user_defined_rules_2/Half.h user_defined_rules_2/c10/util')
        call_subprocess('mv user_defined_rules_2/XPUStream.h user_defined_rules_2/c10/xpu')
        call_subprocess('mv user_defined_rules_2/user_defined_rules_2.cu user_defined_rules_2/src')
    return True

def migrate_test():
    src = []
    extra_args = []
    in_root = os.path.join(os.getcwd(), test_config.current_test)
    test_config.out_root = os.path.join(in_root, 'out_root')
    # Clean the out-root when it exisits.
    if os.path.exists(test_config.out_root):
        shutil.rmtree(test_config.out_root)
    if test_config.current_test == 'cufft_test':
        return do_migrate([os.path.join(in_root, 'cufft_test.cu')], in_root, test_config.out_root, extra_args)

    for dirpath, dirnames, filenames in os.walk(in_root):
        for filename in [f for f in filenames if re.match('.*(cu|cpp|c)$', f)]:
            src.append(os.path.abspath(os.path.join(dirpath, filename)))

    nd_range_bar_exper = ['grid_sync']
    use_masked_sub_group_operation_exper = ['sync_warp_p2', 'asm_shfl_sync_with_exp']
    root_group_exper = ['grid_sync_root_group'] # Current build only.
    logical_group_exper = ['cooperative_groups', 'cooperative_groups_thread_group', 'cooperative_groups_data_manipulate']
    uniform_group_exper = ['cooperative_group_coalesced_group']
    experimental_bfloat16_tests = ['math-experimental-bf16', 'math-experimental-bf162']

    if test_config.current_test in nd_range_bar_exper:
        src.append(' --use-experimental-features=nd_range_barrier ')
    if test_config.current_test in root_group_exper:
        src.append(' --use-experimental-features=root-group ')
    if test_config.current_test == "user_defined_rules":
        src.append(' --rule-file=./user_defined_rules/rules.yaml')
    if test_config.current_test == 'user_defined_rules_2':
        dpct_dir = os.path.dirname(shutil.which("dpct"))
        src.append(' --rule-file=' + dpct_dir + '/../extensions/pytorch_api_rules/pytorch_api.yaml ')
        include_dir = os.path.abspath('user_defined_rules_2')
        src.append(' --extra-arg="-I ' + include_dir + '" ')
        return do_migrate(src, 'user_defined_rules_2/src', test_config.out_root, extra_args)
    if test_config.current_test in logical_group_exper:
        src.append(' --use-experimental-features=logical-group ')
    if test_config.current_test in uniform_group_exper:
        src.append(' --use-experimental-features=non-uniform-groups ')
    if test_config.current_test == 'math_intel_specific':
        src.append(' --rule-file=./math_intel_specific/intel_specific_math.yaml')
    if test_config.current_test.startswith('math-ext-'):
        src.append(' --use-dpcpp-extensions=intel_device_math')
    if test_config.current_test in occupancy_calculation_exper:
        src.append(' --use-experimental-features=occupancy-calculation ')
    if test_config.current_test == 'feature_profiling':
        src.append(' --enable-profiling ')
    if test_config.current_test == 'asm_bar' or test_config.current_test == 'cub_shuffle':
        src.append(' --use-experimental-features=non-uniform-groups ')
    if test_config.current_test == 'cub_block':
        src.append(' --use-experimental-features=user-defined-reductions ')
    if test_config.current_test == 'device_global':
        src.append(' --use-experimental-features=device_global ')
    if test_config.current_test == 'virtual_memory':
        src.append(' --use-experimental-features=virtual_mem ')
    if test_config.current_test == 'in_order_queue_events':
        src.append(' --use-experimental-features=in_order_queue_events ')
    if test_config.current_test in use_masked_sub_group_operation_exper:
        src.append(' --use-experimental-features=masked-sub-group-operation ')
    if test_config.current_test == 'wmma' or test_config.current_test == 'wmma_type':
        src.append(' --use-experimental-features=matrix ')
    if test_config.current_test in experimental_bfloat16_tests:
        src.append(' --use-experimental-features=bfloat16_math_functions ')
    if test_config.current_test == 'const_opt' or test_config.current_test == 'asm_optimize':
        src.append(' --optimize-migration ')
    if test_config.current_test.startswith(('text_experimental_', 'graphics_interop_')) or test_config.current_test == 'driverTex':
        src.append(' --use-experimental-features=bindless_images')
    if "codepin" in test_config.current_test:
        src.append(' --enable-codepin ')
    if test_config.current_test == 'graph':
        src.append(' --use-experimental-features=graph ')
    return do_migrate(src, in_root, test_config.out_root, extra_args)

def manual_fix_for_cufft_external_workspace(migrated_file):
    lines = []
    is_first_occur = True
    with open(migrated_file) as in_f:
        for line in in_f:
            if ('&workSize' in line):
                if (is_first_occur):
                    line = line.replace('&workSize', '&workSize, std::pair(dpct::fft::fft_direction::forward, true)')
                    is_first_occur = False
                else:
                    line = line.replace('&workSize', '&workSize, std::pair(dpct::fft::fft_direction::backward, true)')
            lines.append(line)
    with open(migrated_file, 'w') as out_f:
        for line in lines:
            out_f.write(line)

def manual_fix_for_occupancy_calculation(migrated_file):
    lines = []
    with open(migrated_file) as in_f:
        for line in in_f:
            if ('dpct_placeholder' in line):
                line = line.replace('dpct_placeholder', '0')
            lines.append(line)
    with open(migrated_file, 'w') as out_f:
        for line in lines:
            out_f.write(line)

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    srcs = []
    cmp_options = []
    link_opts = []
    objects = []

    oneDPL_related = ['thrust-vector', 'thrust-for-h2o4gpu', 'thrust-for-RapidCFD', 'cub_device',
             'cub_block_p2']

    oneDNN_related = ['cudnn-activation', 'cudnn-fill', 'cudnn-lrn', 'cudnn-memory',
             'cudnn-pooling', 'cudnn-reorder', 'cudnn-scale', 'cudnn-softmax', 'cudnn-sum', 'cudnn-reduction',
             'cudnn-binary', 'cudnn-bnp1', 'cudnn-bnp2', 'cudnn-bnp3', 'cudnn-normp1', 'cudnn-normp2', 'cudnn-normp3',
             'cudnn-convp1', 'cudnn-convp2', 'cudnn-convp3', 'cudnn-convp4', 'cudnn-convp5', 'cudnn-convp6', 'cudnn-rnn',
             'cudnn-GetErrorString', 'cudnn-convp7',
             'cudnn-types', 'cudnn-version', 'cudnn-dropout', 'matmul', 'matmul_2', 'matmul_3', 'result_type_overload', 'get_library_version'
             ]

    no_fast_math_tests = ['math-emu-half-after11', 'math-emu-half2-after11', 'math-ext-half-after11', 'math-ext-half2-after11',
                          'math-emu-bf16', 'math-emu-bf162', 'math-experimental-bf16', 'math-experimental-bf162']

    if test_config.current_test in oneDPL_related:
        cmp_options.append(prepare_oneDPL_specific_macro())

    if test_config.current_test == 'cub_device_spmv' or re.match('^cu.*', test_config.current_test) or test_config.current_test == 'get_library_version':
        if platform.system() == 'Linux':
            link_opts = test_config.mkl_link_opt_lin
        else:
            link_opts = test_config.mkl_link_opt_win
        cmp_options.append("-DMKL_ILP64")

    if test_config.current_test in no_fast_math_tests:
        cmp_options.append("-fno-fast-math")

    if test_config.current_test.startswith('ccl-'):
        link_opts.append('-lccl -lmpi')

    if "codepin" in test_config.current_test:
        test_config.out_root = test_config.out_root + "_codepin_sycl"

    if test_config.current_test == 'device_global':
        if platform.system() == 'Linux':
            cmp_options.append("-std=c++20")
        else:
            cmp_options.append("-Qstd=c++20")

    if test_config.current_test == 'user_defined_rules_2':
        include_dir = os.path.abspath('.')
        cmp_options.append('-I ' + include_dir + ' ')

    for dirpath, dirnames, filenames in os.walk(test_config.out_root):
        for filename in [f for f in filenames if re.match('.*(cpp|c)$', f)]:
            srcs.append(os.path.abspath(os.path.join(dirpath, filename)))
    if platform.system() == 'Linux':
        link_opts.append(' -lpthread ')
    if test_config.current_test in oneDNN_related:
        if platform.system() == 'Linux':
            link_opts.append(' -ldnnl')
        else:
            link_opts.append(' dnnl.lib')

    if test_config.current_test == 'nvshmem':
        ISHMEMROOT = os.environ['ISHMEMROOT']
        ISHMEMVER = os.environ['ISHMEMVER']

        if (ISHMEMROOT and ISHMEMVER):
            link_opts.append(os.path.join(ISHMEMROOT, ISHMEMVER, 'lib', 'libishmem.a'))

        link_opts.append('-lze_loader -lmpi')

    if (test_config.current_test == 'cufft-external-workspace'):
        manual_fix_for_cufft_external_workspace(srcs[0])
    if (test_config.current_test in occupancy_calculation_exper):
        manual_fix_for_occupancy_calculation(srcs[0])

    ret = False
    if test_config.current_test == 'cufft_test':
        ret = compile_and_link([os.path.join(test_config.out_root, 'cufft_test.dp.cpp')], cmp_options, objects, link_opts)
    elif re.match('^cufft.*', test_config.current_test) and platform.system() == 'Linux':
        ret = compile_and_link(srcs, cmp_options, objects, link_opts)
    elif test_config.current_test in cmpl_only_tests:
        ret = compile_files(srcs, cmp_options)
    else:
        ret = compile_and_link(srcs, cmp_options, objects, link_opts)
    return ret


def run_test():
    if test_config.current_test in cmpl_only_tests or test_config.current_test in cmpllink_only_as_wa_tests:
        return True
    if test_config.current_test.startswith(('text_experimental_obj_', 'text_experimental_tex_', 'graphics_interop_')) and test_config.device_filter.count("cuda") == 0:
        return True
    os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
    os.environ['CL_CONFIG_CPU_EXPERIMENTAL_FP16']="1"
    if test_config.current_test.startswith('ccl-test'):
        return call_subprocess('mpirun -n 2 ' + os.path.join(os.path.curdir, test_config.current_test + '.run '))
    return run_binary_with_args()
