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

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

from test_utils import *

exec_tests = ['thrust-vector-2', 'thrust-binary-search', 'thrust-count', 'thrust-copy',
              'thrust-qmc', 'thrust-transform-if', 'thrust-policy', 'thrust-list', 'module-kernel',
              'kernel-launch', 'thrust-gather', 'thrust-scatter', 'thrust-unique_by_key_copy', 'thrust-for-hypre',
              'thrust-rawptr-noneusm', 'driverStreamAndEvent', 'grid_sync', 'deviceProp', 'gridThreads', 'kernel_library', 'cub_block_p2',
              'cub_constant_iterator', 'cub_device_reduce_max', 'cub_device_reduce_min', 'cub_discard_iterator', 'ccl', 'ccl_test2'
              'cub_device', 'cub_device_reduce_sum', 'cub_device_reduce', 'cub_device_reduce_by_key', 'cub_device_select_unique_by_key',
              'cub_device_scan_inclusive_scan', 'cub_device_scan_exclusive_scan', 'cub_device_seg_radix_sort_pairs',
              'cub_device_scan_inclusive_sum', 'cub_device_scan_exclusive_sum', 'cub_device_select_unique', 'cub_device_radix_sort_keys', 'cub_device_radix_sort_pairs',
              'cub_device_select_flagged', 'cub_device_run_length_encide_encode', 'cub_counting_iterator', 'cub_arg_index_input_iterator',
              'cub_device_inclusive_sum_by_key', 'cub_device_exclusive_sum_by_key', 'cub_device_inclusive_scan_by_key', 'cub_device_exclusive_scan_by_key',
              'cub_device_reduce_arg', 'cub_device_seg_sort_pairs', 'cub_intrinsic',
              'cub_transform_iterator', 'activemask', 'complex', 'thrust-math', 'libcu_array', 'libcu_complex', 'libcu_tuple',
              'user_defined_rules', 'math-exec', 'math-habs', 'math-emu-double', 'math-emu-float', 'math-emu-half', 'math-emu-half2', 'math-emu-simd',
              'math-ext-double', 'math-ext-float', 'math-ext-half', 'math-ext-half2', 'math-ext-simd', 'cudnn-activation',
              'cudnn-fill', 'cudnn-lrn', 'cudnn-memory', 'cudnn-pooling', 'cudnn-reorder', 'cudnn-scale', 'cudnn-softmax',
              'cudnn-sum', 'math-funnelshift', 'thrust-sort_by_key', 'thrust-find', 'thrust-inner_product', 'thrust-reduce_by_key',
              'math-bf16-conv',
              'math-bfloat16', 'libcu_atomic', 'test_shared_memory', 'cudnn-reduction', 'cudnn-binary', 'cudnn-bnp1', 'cudnn-bnp2', 'cudnn-bnp3',
              'cudnn-normp1', 'cudnn-normp2', 'cudnn-normp3', 'cudnn-convp1', 'cudnn-convp2', 'cudnn-convp3', 'cudnn-convp4', 'cudnn-convp5',
              'cudnn_mutilple_files', "cusparse_1", "cusparse_2",
              'cudnn-GetErrorString',
              'cudnn-types', 'cudnn-version', 'cudnn-dropout',
              'constant_attr', 'sync_warp_p2',
              'thrust-unique_by_key', 'cufft_test', 'cufft-external-workspace', "pointer_attributes", 'math_intel_specific', 'math-drcp', 'thrust-pinned-allocator', 'driverMem',
              'cusolver_test1', 'cusolver_test2', 'cusolver_test3', 'thrust_op', 'cublas-extension', 'cublas_v1_runable', 'thrust_minmax_element',
              'thrust_is_sorted', 'thrust_partition', 'thrust_remove_copy', 'thrust_unique_copy', 'thrust_transform_exclusive_scan',
              'thrust_set_difference', 'thrust_set_difference_by_key', 'thrust_set_intersection_by_key', 'thrust_stable_sort',
              'thrust_tabulate', 'thrust_for_each_n', 'device_info', 'defaultStream', 'cudnn-rnn', 'feature_profiling',
              'thrust_raw_reference_cast', 'thrust_partition_copy', 'thrust_stable_partition_copy',
              'thrust_stable_partition', 'thrust_remove', 'cub_device_segmented_sort_pairs', 'thrust_find_if_not',
              'thrust_find_if', 'thrust_mismatch', 'thrust_replace_copy', 'thrust_reverse',
              'cooperative_groups']


def setup_test():
    return True

def migrate_test():
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

    math_extension_tests = ['math-ext-double', 'math-ext-float', 'math-ext-half', 'math-ext-half2', 'math-ext-simd']

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
    if test_config.current_test in math_extension_tests:
        src.append(' --use-dpcpp-extensions=intel_device_math')
    if test_config.current_test == 'feature_profiling':
        src.append(' --enable-profiling ')
    if test_config.current_test == 'sync_warp_p2':
        src.append(' --use-experimental-features=masked_sub_group_function ')
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
             'cudnn-convp1', 'cudnn-convp2', 'cudnn-convp3', 'cudnn-convp4', 'cudnn-convp5', 'cudnn-rnn',
             'cudnn-GetErrorString',
             'cudnn-types', 'cudnn-version', 'cudnn-dropout'
             ]

    if test_config.current_test in oneDPL_related:
        cmp_options.append(prepare_oneDPL_specific_macro())

    if re.match('^cu.*', test_config.current_test):
        if platform.system() == 'Linux':
            link_opts = test_config.mkl_link_opt_lin
        else:
            link_opts = test_config.mkl_link_opt_win
        cmp_options.append("-DMKL_ILP64")

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

    if (test_config.current_test == 'cufft-external-workspace'):
        manual_fix_for_cufft_external_workspace(srcs[0])

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
    if test_config.current_test not in exec_tests:
        return True
    os.environ['ONEAPI_DEVICE_SELECTOR'] = test_config.device_filter
    os.environ['CL_CONFIG_CPU_EXPERIMENTAL_FP16']="1"
    if test_config.current_test == 'ccl':
        return call_subprocess('mpirun -n 2 ' + os.path.join(os.path.curdir, test_config.current_test + '.run '))
    return run_binary_with_args()

