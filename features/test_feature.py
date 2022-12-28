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

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

exec_tests = ['thrust-vector-2', 'thrust-binary-search', 'thrust-count', 'thrust-copy',
              'thrust-qmc', 'thrust-transform-if', 'thrust-policy', 'thrust-list', 'module-kernel',
              'kernel-launch', 'thrust-gather', 'thrust-scatter', 'thrust-unique_by_key_copy', 'thrust-for-hypre',
              'thrust-rawptr-noneusm', 'driverStreamAndEvent', 'grid_sync', 'deviceProp', 'gridThreads', 'cub_block_p2',
              'cub_constant_iterator',
              'cub_device', 'cub_device_reduce_sum', 'cub_device_reduce', 'cub_device_reduce_by_key',
              'cub_device_scan_inclusive_scan', 'cub_device_scan_exclusive_scan',
              'cub_device_scan_inclusive_sum', 'cub_device_scan_exclusive_sum', 'cub_device_select_unique',
              'cub_device_select_flagged', 'cub_device_run_length_encide_encode', 'cub_counting_iterator',
              'cub_transform_iterator', 'activemask', 'complex', 'thrust-math'
              'user_defined_rules', 'math-exec', 'math-habs', 'cudnn-activation',
              'cudnn-fill', 'cudnn-lrn', 'cudnn-memory', 'cudnn-pooling', 'cudnn-reorder', 'cudnn-scale', 'cudnn-softmax',
              'cudnn-sum', 'math-funnelshift', 'ccl', 'thrust-sort_by_key', 'thrust-find', 'thrust-inner_product', 'thrust-reduce_by_key',
              'math-bfloat16', 'libcu_atomic', 'test_shared_memory', 'cudnn-reduction', 'cudnn-binary', 'cudnn-bnp1', 'cudnn-bnp2', 'cudnn-bnp3',
              'cudnn-normp1', 'cudnn-normp2', 'cudnn-normp3', 'cudnn-convp1', 'cudnn-convp2', 'cudnn-convp3', 'cudnn-convp4', 'cudnn-convp5',
              'thrust-unique_by_key', 'cufft_test', "pointer_attributes", 'math_intel_specific', 'math-drcp', 'thrust-pinned-allocator', 'driverMem',
              'cusolver_test1', 'cusolver_test2', 'thrust_op', 'cublas-extension', 'cublas_v1_runable', 'thrust_minmax_element',
              'thrust_is_sorted', 'thrust_partition', 'thrust_remove_copy', 'thrust_unique_copy']


def setup_test():
    return True

def migrate_test():
    print("migrate_test:", test_config.current_test)
    src = []
    extra_args = []
    in_root = os.path.join(os.getcwd(), test_config.current_test)
    test_config.out_root = os.path.join(in_root, 'out_root')

    if test_config.current_test == 'cufft_test':
        return do_migrate([os.path.join(in_root, 'cufft_test.cu')], in_root, test_config.out_root, extra_args)

    for dirpath, dirnames, filenames in os.walk(in_root):
        for filename in [f for f in filenames if re.match('.*(cu|cpp|c)$', f)]:
            src.append(os.path.abspath(os.path.join(dirpath, filename)))

    # if 'module-kernel' in current_test:
    size_deallocation = ['DplExtrasAlgorithm_api_test7', 'DplExtrasAlgorithm_api_test8',
                        'DplExtrasVector_api_test1', 'DplExtrasVector_api_test2']
    nd_range_bar_exper = ['grid_sync', 'Util_api_test12']
    logical_group_exper = ['cooperative_groups', 'Util_api_test23', 'Util_api_test24', 'Util_api_test25']

    if test_config.current_test in size_deallocation:
        extra_args.append(' -fsized-deallocation ')
    if test_config.current_test in nd_range_bar_exper:
        src.append(' --use-experimental-features=nd_range_barrier ')
    if test_config.current_test == "user_defined_rules":
        src.append(' --rule-file=./user_defined_rules/rules.yaml')
    if test_config.current_test in logical_group_exper:
        src.append(' --use-experimental-features=logical-group ')
    if test_config.current_test == 'math_intel_specific':
        src.append(' --rule-file=./math_intel_specific/intel_specific_math.yaml')

    return do_migrate(src, in_root, test_config.out_root, extra_args)

def build_test():
    if (os.path.exists(test_config.current_test)):
        os.chdir(test_config.current_test)
    srcs = []
    cmp_options = []
    link_opts = []
    objects = []

    oneDPL_related = ['thrust-vector', 'thrust-for-h2o4gpu', 'thrust-for-RapidCFD', 'cub_device',
             'cub_block_p2', 'DplExtrasDpcppExtensions_api_test1', 'DplExtrasDpcppExtensions_api_test2',
             'DplExtrasDpcppExtensions_api_test3', 'DplExtrasDpcppExtensions_api_test4']

    oneDNN_related = ['cudnn-activation', 'cudnn-fill', 'cudnn-lrn', 'cudnn-memory',
             'cudnn-pooling', 'cudnn-reorder', 'cudnn-scale', 'cudnn-softmax', 'cudnn-sum', 'cudnn-reduction',
             'cudnn-binary', 'cudnn-bnp1', 'cudnn-bnp2', 'cudnn-bnp3', 'cudnn-normp1', 'cudnn-normp2', 'cudnn-normp3',
             'cudnn-convp1', 'cudnn-convp2', 'cudnn-convp3', 'cudnn-convp4', 'cudnn-convp5']

    if test_config.current_test in oneDPL_related:
        cmp_options.append(prepare_oneDPL_specific_macro())

    if re.match('^cu.*', test_config.current_test):
        if test_config.current_test not in oneDNN_related:
            if platform.system() == 'Linux':
                link_opts = test_config.mkl_link_opt_lin
            else:
                link_opts = test_config.mkl_link_opt_win

    if test_config.current_test == 'ccl':
        link_opts.append('-lccl -lmpi')

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
    ret = False
    print("test_config.current_test:", test_config.current_test)
    if test_config.current_test == 'cufft_test':
        ret = compile_and_link([os.path.join(test_config.out_root, 'cufft_test.dp.cpp')], cmp_options, objects, link_opts)
    elif test_config.current_test in exec_tests:
        ret = compile_and_link(srcs, cmp_options, objects, link_opts)
    elif re.match('^cufft.*', test_config.current_test) and platform.system() == 'Linux':
        ret = compile_and_link(srcs, cmp_options, objects, link_opts)
    else:
        ret = compile_files(srcs, cmp_options)
    return ret


def run_test():
    print("run_test:", test_config.current_test)
    if test_config.current_test not in exec_tests:
        return True
    os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
    os.environ['CL_CONFIG_CPU_EXPERIMENTAL_FP16']="1"
    if test_config.current_test == 'ccl':
        return call_subprocess('mpirun -n 2 ' + os.path.join(os.path.curdir, test_config.current_test + '.run '))
    return run_binary_with_args()

