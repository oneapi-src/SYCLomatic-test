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

1. This sample project contains a simple CUDA program with 12 files:

```
CUDA
├── CMakeLists.txt
├── cudaImage.cu
├── cudaImage.h
├── cudaSift.h
├── cudaSiftD.cu
├── cudaSiftD.h
├── cudaSiftH.cu
├── cudaSiftH.h
├── cudautils.h
├── geomFuncs.cpp
├── mainSift.cpp
└── matching.cu
```
2. Make sure the ```OpenCV*``` is installed on the machine. ```
$ sudo apt-get install libopencv-dev
```
Then, make a `build` directory to use the **cmake** command line tool to generate the corresponding build tool (make) directly.
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
├── cudaImage.dp.cpp
├── cudaImage.h
├── cudaSift.h
├── cudaSift.h.yaml
├── cudaSiftD.dp.cpp
├── cudaSiftD.h
├── cudaSiftH.dp.cpp
├── cudaSiftH.h
├── cudautils.h
├── cudautils.h.yaml
├── geomFuncs.cpp
├── mainSift.cpp.dp.cpp
└── matching.dp.cpp

```

3. Inspect the migrated source code, address any `DPCT` warnings generated
   by the Intel® DPC++ Compatibility Tool, and verify the new program correctness.

Warnings are printed to the console and added as comments in the migrated
source. See *Diagnostic Reference* in the [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) for more information on what each warning means.


This sample should generate the following warnings:
```
warning: #DPCT2001:228: You can link with more library by add them here.
LIB :=  
```


See below **Addressing Warnings in the Migrated Code** to understand how to resolve the warning.


4. Build the migrated code with generated Makefile.dpct
```
$ cd out
$ make -f Makefile.dpct
# Please make sure the oneAPI package was installed before building the application to make sure the oneAPI DPC++ compiler was installed.
```

# Addressing Warnings in Migrated Code

Migration generated one warning for code that `dpct` could not migrate:
```
warning: #DPCT2001:228: You can link with more library by add them here.
LIB :=  
```
This message is shown in the Makefile.dpct, for **cudaSift** the application need to link the **OpenCV** libraries during the link time. Modifing the Makefile.dpct will fix the linker error.
```
LIB :=  -lopencv_core -lopencv_imgcodecs
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
$ ./cudasift 
Image size = (1920,1080)
Initializing data...
Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of Points after sift extraction =  3681

Number of Points after sift extraction =  3933

Number of original features: 3681 3933
Number of matching features: 1220 1258 33.1432% 1 2

Performing data verification 
Data verification is SUCCESSFUL. 

Total workload time = 2206.28 ms
```
**Note:** The testing result was running on Intel(R) Core(TM) i7-13700K on the CPU backend with 2023.2 oneAPI released oneAPI packaged. 

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
See
[License.txt](https://github.com/oneapi-src/Velocity-Bench/blob/main/cudaSift/LICENSE.md) for details.
