# SYCLomatic Tool: Migrate bitcracker APP
## Use the command line to migrate large code base.
The SYCLomatic project (the Open source version of Intel® DPC++ Compatibility Tool) can migrate project that contain multiple source and header files. 
| Optimized for         | Description
|:---                   |:---
| OS                    | Linux* Ubuntu* 22.04
| Software              | Intel® DPC++ Compatibility Tool
| What you will learn   | Simple invocation of dpct to migrate CUDA code
| Time to complete      | 15 minutes


# Purpose
The SYCLomatic tool can migrate projects composed with multiple source and header files.
Used the dpct option **--in-root** option to set the root location of your prepared migration APP. Only the files under this specified root will be considered to migrate. Files located outside the **--in-root** will be considered system files or libraries files and will not be migrated. 

The dpct **--out-root** will specify the directory into which generated SYCL*-compilant code producted by the dpct tool is written. The relative path and the name will be kept, except the file extensions are changed to **.dp.cpp**.


# Key Implementation Details
Except the --in-root and --out-root options, there are additional options can help to migrate the code more smoothly: [Command Line Options Reference](https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/command-line-options-reference.html).



## Migrating the CUDA Sample to Data Parallel C++ with the Intel® DPC++ Compatibility Tool

Building and running the CUDA sample is not required to migrate this project
to a SYCL*-compliant project.

> **Note**: Certain CUDA header files, referenced by the CUDA application
> source files to be migrated, need to be accessible for the migration step.
> See *Before you Begin* in [Get Started with the Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-dpcpp-compatibility-tool/top.html#top_BEFORE_YOU_BEGIN).

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script located in
> the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: `. ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `$ bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - For Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or MacOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).


### Command-Line on a Linux* System

1. This sample project contains a simple CUDA program with eight files
   (CUDA/CMakeLists.txt, CUDA/src/aes.h, CUDA/src/attack.cu, CUDA/src/bitcracker.h, CUDA/src/main.cu, CUDA/src/sha256.h, CUDA/src/utils.cu and CUDA/src/w_blocks.cu) located in CUDA directory and the sub-directory src of CUDA:

```
CUDA
├── CMakeLists.txt
└── src
    ├── aes.h
    ├── attack.cu
    ├── bitcracker.h
    ├── main.cu
    ├── sha256.h
    ├── utils.cu
    └── w_blocks.cu
```
2. Make a `build` directory to use the **cmake** command line tool to generate the corresponding build tool (make) directly.
```sh
$ cd CUDA && mkdir build
$ cd build && cmake ..
```
3. Use the **intercept-build** tool to intercept the build step to generate the compilation database `compile_commands.json` file under the same fodler.
``` sh
$ intercept-build make
$ ls .
CMakeCache.txt  CMakeFiles  Makefile  bitcracker  cmake_install.cmake  compile_commands.json
```
2. Use the tool's `--in-root` option and provide input files to specify where
   to locate the CUDA files that needs migration; use the tool’s `--out-root`
   option to designate where to generate the resulting files(default is `dpct_output`); use the tool's `-p` option to specify compilation database to migrate the whole project:

```sh
# From the CUDA directory as root directory:
$ cd ..
$ dpct --in-root=. -p=./build/compile_commands.json --out-root=out --gen-build-script --cuda-include-path=/usr/local/cuda/include
```

> If an `--in-root` option is not specified, the directory of the first input
> source file is implied. If `--out-root` is not specified, `./dpct_output`
> is implied.

You should see the migrated files in the `out` folder that was specified
by the `--out-root` option:

```
out/
├── MainSourceFiles.yaml
├── Makefile.dpct
└── src
    ├── aes.h
    ├── aes.h.yaml
    ├── attack.dp.cpp
    ├── bitcracker.h
    ├── bitcracker.h.yaml
    ├── main.dp.cpp
    ├── sha256.h
    ├── sha256.h.yaml
    ├── utils.dp.cpp
    └── w_blocks.dp.cpp
```

3. Inspect the migrated source code, address any `DPCT` warnings generated
   by the Intel® DPC++ Compatibility Tool, and verify the new program correctness.

Warnings are printed to the console and added as comments in the migrated
source. See *Diagnostic Reference* in the [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) for more information on what each warning means.


This sample should generate the following warnings:
```
warning: *DPCT1110:0*: The total declared local variable size in device function
decrypt_vmk_with_mac exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
```

```
warning: DPCT1009:0: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
```

See below **Addressing Warnings in the Migrated Code** to understand how to
resolve the warning.


4. Build the migrated code with generated Makefile.dpct
```
$ cd out
$ make -f Makefile.dpct
# Please make sure the oneAPI package was installed before building the application to make sure the oneAPI DPC++ compiler was installed.
```

# Addressing Warnings in Migrated Code

Migration generated one warning for code that `dpct` could not migrate:
```
warning: *DPCT1110:0*: The total declared local variable size in device function
decrypt_vmk_with_mac exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
```
This message is shown because the Compatibility Tool migrated find the user declared private memeory size of local variable in the kernel will exceed the 128 bytes which is the largest register size for the each work-item on the Intel XE core when the sub-group size is 32.

Open **out/src/attack.dp.cpp** and find the error **DPCT1110**, the application defined 56 **uint32_t** type value, totally need 224 bytes private value which exceed the 128 bytes on the XE GPU vector engine register size. The migrated code didn't specify the sub group size, let compiler to determine the size. And user can explicitly specify the sub group size to 16 by ```[[intel::reqd_sub_group_size(16)]]``` after the submit function.

```
          cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(16)]] {
                    decrypt_vmk_with_mac(
                        num_read_pswd, d_found, d_vmk, d_vmkIV, d_mac, d_macIV,
                        d_computedMacIV, v0, v1, v2, v3, s0, s1, s2, s3,
                        d_pswd_uint32, d_w_words_uint32, item_ct1, TS0_ptr_ct1,
                        TS1_ptr_ct1, TS2_ptr_ct1, TS3_ptr_ct1);
                });
```


```
warning: DPCT1009:5: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
```

As you have noticed, the migration of this project resulted in one DPCT
message that needs to be addressed, **DPCT1009**. This message is shown because 
the Compatibility Tool migrated from returning an error code in the CUDA code to determine whether the CUDA execution was successful or not, but the SYCL used the try-exception to catch the failure of the API call. manually adjusting is needed to generate the SYCL compliant code.

Open out/src/bitcracker.h and locate the error **DPCT1009**. Then make the
following changes:

Remove the macro definition:
```
#define CUDA_CHECK(call)                                                       \
   { dpct::err0 err = call; }
```

You’ll also need to change the macro expansion for all the files in the **out** directory and sub-directory.

Strip the CUDA_CHECK macro expansion under the **out** folder:
```
./src/bitcracker.h:115:#define CUDA_CHECK(call)                                                       \
./src/attack.dp.cpp:941:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:948:    CUDA_CHECK(DPCT_CHECK_ERROR(h_pswd_char = sycl::malloc_host<char>(
./src/attack.dp.cpp:956:    CUDA_CHECK(DPCT_CHECK_ERROR(h_pswd_uint32 = sycl::malloc_host<uint32_t>(
./src/attack.dp.cpp:968:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:975:    CUDA_CHECK(
./src/attack.dp.cpp:978:    CUDA_CHECK(DPCT_CHECK_ERROR(d_vmkIV = sycl::malloc_device<uint8_t>(
./src/attack.dp.cpp:980:    CUDA_CHECK(DPCT_CHECK_ERROR(d_mac = sycl::malloc_device<uint8_t>(
./src/attack.dp.cpp:982:    CUDA_CHECK(DPCT_CHECK_ERROR(d_macIV = sycl::malloc_device<uint8_t>(
./src/attack.dp.cpp:984:    CUDA_CHECK(DPCT_CHECK_ERROR(d_computedMacIV = sycl::malloc_device<uint8_t>(
./src/attack.dp.cpp:986:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:989:    CUDA_CHECK(DPCT_CHECK_ERROR(d_pswd_uint32 = sycl::malloc_device<uint32_t>(
./src/attack.dp.cpp:994:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:998:    CUDA_CHECK(
./src/attack.dp.cpp:1002:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1006:    CUDA_CHECK(
./src/attack.dp.cpp:1010:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1014:    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
./src/attack.dp.cpp:1020:    CUDA_CHECK(
./src/attack.dp.cpp:1068:        CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1073:        CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1115:                CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1126:                CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1130:                CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1175:    CUDA_CHECK(
./src/attack.dp.cpp:1177:    CUDA_CHECK(
./src/attack.dp.cpp:1179:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1182:    CUDA_CHECK(DPCT_CHECK_ERROR(sycl::free(d_vmk, dpct::get_in_order_queue())));
./src/attack.dp.cpp:1183:    CUDA_CHECK(
./src/attack.dp.cpp:1185:    CUDA_CHECK(DPCT_CHECK_ERROR(sycl::free(d_mac, dpct::get_in_order_queue())));
./src/attack.dp.cpp:1186:    CUDA_CHECK(
./src/attack.dp.cpp:1188:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/attack.dp.cpp:1190:    CUDA_CHECK(
./src/attack.dp.cpp:1192:    CUDA_CHECK(DPCT_CHECK_ERROR(
./src/w_blocks.dp.cpp:202:        CUDA_CHECK(
./src/w_blocks.dp.cpp:205:        CUDA_CHECK(
./src/w_blocks.dp.cpp:210:        CUDA_CHECK(DPCT_CHECK_ERROR(
./src/w_blocks.dp.cpp:214:        CUDA_CHECK(
./src/w_blocks.dp.cpp:235:    CUDA_CHECK(
./src/w_blocks.dp.cpp:247:    CUDA_CHECK(DPCT_CHECK_ERROR(sycl::free(salt_d, dpct::get_in_order_queue())));
./src/w_blocks.dp.cpp:248:    CUDA_CHECK(
./src/main.dp.cpp:193:        CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
./src/main.dp.cpp:205:        CUDA_CHECK(
./src/main.dp.cpp:234:            CUDA_CHECK(DPCT_CHECK_ERROR(
```
## Rebuild the migrated code
After manually addressing the warning error, need to rebuild the application.
```
$ make -f Makefile.dpct clean
$ make -f Makefile.dpct 
```
# Example Output

When you run the migrated application, you should see the following console
output:

```
$: ./bitcracker -f ../../hash_pass/img_win8_user_hash.txt -d ../../hash_pass/user_passwords_60000.txt -b 60000
---------> BitCracker: BitLocker password cracking tool <---------


==================================
Retrieving Info
==================================

Reading hash file "../../hash_pass/img_win8_user_hash.txt"
================================================
                  Attack
================================================
Type of attack: User Password
Psw per thread: 1
max_num_pswd_per_read: 60000
Dictionary: ../../hash_pass/user_passwords_60000.txt
MAC Comparison (-m): Yes


Iter: 1, num passwords read: 60000
Kernel execution:
        Effective passwords: 60000
        Passwords Range:
                npknpByH7N2m3OnLNH1X9DJxLrzIFWk
                .....
                dL_7uuf3QCz-c6K3xDu0
--------------------
================================================
Bitcracker attack completed
Total passwords evaluated: 60000
Password not found!
================================================
time to subtract from total: 0.0148924 s
bitcracker - total time for whole calculation: 452.283 s
```
**Note:** The testing result was running on Intel(R) Core(TM) i7-13700K on the CPU backend with 2023.2 oneAPI released oneAPI packaged. 

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the GNU General Public License version 2. See
[License.txt](https://github.com/oneapi-src/Velocity-Bench/blob/main/bitcracker/LICENSE.md) for details.
