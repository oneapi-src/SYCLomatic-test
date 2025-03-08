# Migrate ktransformers to SYCL version
[SYCLomatic](https://github.com/oneapi-src/SYCLomatic) is a project to assist developers in migrating their existing code written in different programming languages to the SYCL* C++ heterogeneous programming model. It is an open source version of the Intel® DPC++ Compatibility Tool.

This file lists the detailed steps to migrate CUDA version of [ktransformers](https://github.com/kvcache-ai/ktransformers.git) to SYCL version with SYCLomatic. As follow table summarizes the migration environment, the software required, and so on.

   | Optimized for         | Description
   |:---                   |:---
   | OS                    | Linux* Ubuntu* 22.04
   | Software              | Intel® oneAPI Base Toolkit, SYCLomatic
   | What you will learn   | Migration of CUDA code, Run SYCL code on oneAPI and Intel device
   | Time to complete      | TBD


## Migrating ktransformers to SYCL

### 1 Prepare the migration
#### 1.1 Get the source code of ktransformers and install the dependencies
```sh
   $ git clone https://github.com/kvcache-ai/ktransformers.git
   $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   $ export PATH=/usr/local/cuda:$PATH
   $ export PATH=/usr/local/cuda-12.4/bin:$PATH
```

#### 1.2 Prepare migration tool and environment

 * Install SYCL run environment [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html). After installation, the Intel® DPC++ Compatibility tool is also available, set up the SYCL run environment as follows:

```
   $ source /opt/intel/oneapi/setvars.sh
   $ dpct --version  # Intel® DPC++ Compatibility tool version
```
 * If want to try the latest version of the compatibility tool, try to install SYCLomatic by downloading prebuild of [SYCLomatic release](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/README.md#Releases) or [build from source](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/README.md), as follow give the steps to install prebuild version:
 ```
   $ export SYCLomatic_HOME=/path/to/install/SYCLomatic
   $ mkdir $SYCLomatic_HOME
   $ cd $SYCLomatic_HOME
   $ wget https://github.com/oneapi-src/SYCLomatic/releases/download/20240203/linux_release.tgz   #Change the timestamp 20240203 to latest one
   $ tar xzvf linux_release.tgz
   $ source setvars.sh
   $ dpct --version #SYCLomatic version
 ```

For more information on configuring environment variables, see [Use the setvars Script with Linux*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### 2 Migrate the source code
Here, we use [custom_gguf](https://github.com/kvcache-ai/ktransformers/tree/main/ktransformers/ktransformers_ext/cuda/custom_gguf) as an example to explain the migrate process.

```sh
# custom_gguf_HOME=ktransformers/ktransformers/ktransformers_ext/cuda/custom_gguf/
$ export PATH_TO_C2S_INSTALL_FOLDER=~/workspace/c2s_install
$ source $PATH_TO_C2S_INSTALL_FOLDER/setvars.sh
$ cd ${custom_gguf_HOME}
$ c2s dequant.cu \
   --extra-arg="-I/~/.local/lib/python3.10/site-packages/torch/include" \
   --extra-arg="-I/~/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include" \
   --extra-arg="-I/usr/include/python3.10" \
   --rule-file=~/workspace/c2s_install/extensions/pytorch_api_rules/pytorch_api.yaml
```

Now you can see the migrated files in ${custom_gguf_HOME}/dpct_output.

### 3 Prepare the running environment
#### 3.1 Create virtual environment and source oneapi
```
$ python3 -m venv xputorch
$ source ~/workspace/xputorch/bin/activate
$ source /opt/intel/oneapi/setvars.sh
$ export LD_LIBRARY_PATH=~/workspace/xputorch/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```
#### 3.2 Install xpu torch
Install xpu torch through 

```
pip install torch==2.7.0.dev20250305+xpu --extra-index-url https://download.pytorch.org/whl/nightly/xpu
```

### 4 Build the migrated ktransformers
There serveral tests available in the current stage:
* 3 sycl tests to test single kernel (passed) in ./migrated/single_kernel_test
* 4 sycl tests to test single kernel (results mismatch) in ./migrated/single_kernel_test_need_debug
* 1 torch test to test dequantize_q8_0 in ./migrated/torch_test  
* 9 pytorch test to test in ./migrated/python_test, passed with random generated input, compared with cpu calculation
  * dequantize_f32
  * dequantize_f16
  * dequantize_q8_0
  * dequantize_q2_k
  * dequantize_q3_k
  * dequantize_q4_k
  * dequantize_q5_k
  * dequantize_q6_k
  * dequantize_iq4_xs

For the c++ test, you can select one - ${test_directory}/${test_name}, and compile it through
```
$ cd ${test_directory}
$ source /opt/intel/oneapi/setvars.sh
$ icpx -fsycl -I/opt/intel/oneapi/compiler/latest/include/sycl -I/~/workspace/xputorch/lib/python3.10/site-packages/torch/include -I/usr/include/python3.10 -I/~/workspace/xputorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -L/~/workspace/xputorch/lib/python3.10/site-packages/torch/lib -ltorch_xpu -ltorch_cpu -lc10_xpu -lc10 ${test_name} -o ${out_name}
```

For the python test, you need to bulid extension and run python test
```
$ source ~/workspace/xputorch/bin/activate
$ source /opt/intel/oneapi/setvars.sh
$ unset CPATH  # avoid duplicated headers in pytorch sycl
$ cd third-party-programs/ktransformers/custom_gguf/migrated
$ export CC=icpx
$ export CXX=icpx
$ source  $SYCLomatic_HOME/setvars.sh
$ python3 setup.py build_ext --inplace

# Run the pytest
$ pip install pytest
$ cd python_test
$ ptest test_dequant.py 
```

### 5 Run migrated SYCL version ktransformers
```
$ ./${out_name}
```


## ktransformers License
[LICENSE](https://github.com/kvcache-ai/ktransformers/blob/main/LICENSE)

## Reference
* Command Line Options of [SYCLomatic](https://oneapi-src.github.io/SYCLomatic/dev_guide/command-line-options-reference.html) or [Intel® DPC++ Compatibility Tool](https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/command-line-options-reference.html)
* [oneAPI GPU Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/)
* [SYCLomatic project](https://github.com/oneapi-src/SYCLomatic/)


## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.
\*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
