# SYCLomatic Tool: Migrate hplinpack APP
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

1. This sample project contains a simple CUDA program, located in ```cuda``` directory and the sub-directory src of ```cuda```:

2. Use the **intercept-build** tool to intercept the build step to generate the compilation database `compile_commands.json` file under the same fodler.
``` sh
$ git clone https://github.com/oneapi-src/Velocity-Bench.git
$ cd hplinpack/cuda/hp-2.3
$ intercept-build make
```
2. Use the tool's `--in-root` option and provide input files to specify where
   to locate the CUDA files that needs migration; use the tool’s `--out-root`
   option to designate where to generate the resulting files(default is `dpct_output`); use the tool's `-p` option to specify compilation database to migrate the whole project and use the `--gen-build-script` to generate the `Makefile.dpct` for the migrated code:

```sh
# From the cuda directory as root directory:
$ dpct --in-root=. --out-root=out --cuda-include-path=/usr/local/cuda/include -p . --gen-build-script
```

> If an `--in-root` option is not specified, the directory of the first input
> source file is implied. If `--out-root` is not specified, `./dpct_output`
> is implied.

You should see the migrated files in the `out` folder that was specified
by the `--out-root` option.

3. To build the migration app, the Makefile.dpct needs to be updated. Details are in the following:


```make
5 #DPCT2001:4: You can link with more library by add them here.
6 LIB :=
7
8 FLAGS :=
9
......
582 TARGET :=  ${TARGET_0} ${TARGET_1} ${TARGET_2}
......
589 $(TARGET_0): $(OBJS_0)
590         $(CC) -fsycl -o $@ $^ $(LIB) -qmkl
......
628 $(TARGET_1): $(OBJS_1)
629         ar -r $@ $^ $(LIB) -qmkl
......
1009 $(TARGET_2): $(OBJS_2)
1010         $(CC) -fsycl -o $@ $^ $(LIB) -qmkl
1011
1012 $(TARGET_2_OBJ_0):$(TARGET_2_SRC_0)
1013         cc -c ${TARGET_2_SRC_0} -o ${TARGET_2_OBJ_0} $(TARGET_2_FLAG_0)
1014
1015 $(TARGET_2_OBJ_1):$(TARGET_2_SRC_1)
1016         cc -c ${TARGET_2_SRC_1} -o ${TARGET_2_OBJ_1} $(TARGET_2_FLAG_1)
```
change to 
``` make
5 #DPCT2001:4: You can link with more library by add them here.
6 LIB := -lmpi
7
8 FLAGS := -fPIC
9
......
582 TARGET :=   ${TARGET_1} ${TARGET_2} ${TARGET_0}
......
589 $(TARGET_0): $(OBJS_0)
590         $(CC) -fsycl -o $@ $^ $(LIB) -qmkl libdgemm.so.1.0.1 ../lib/intel64/libhpl.a
627
628 $(TARGET_1): $(OBJS_1)
629         ar -r $@ $^ $(LIB)
630
1008
1009 $(TARGET_2): $(OBJS_2)
1010         $(CC) -fPIC -shared -fsycl -o $@ $^ $(LIB) -qmkl
1011
1012 $(TARGET_2_OBJ_0):$(TARGET_2_SRC_0)
1013         cc -c ${TARGET_2_SRC_0} -o ${TARGET_2_OBJ_0} $(TARGET_2_FLAG_0)
1014
1015 $(TARGET_2_OBJ_1):$(TARGET_2_SRC_1)
1016         icpx -c  ${TARGET_2_SRC_1} -o ${TARGET_2_OBJ_1} $(TARGET_2_FLAG_1)
1017
```
run the command ```vimdiff Makefile.dpct Makefile.dpct.patched``` in the out folder can get the changing details.


4. Build the migrated code with generated Makefile.dpct
```
$ make -f Makefile.dpct
# Please make sure the oneAPI package was installed before building the application to make sure the oneAPI DPC++ compiler was installed.
```

# Example Output

When you run the migrated application, you can follow the [README](https://github.com/oneapi-src/Velocity-Bench/blob/main/hplinpack/README.md)

The output of hplinpack
```
================================================================================
HPLinpack 2.3  --  High-Performance Linpack benchmark  --   December 2, 2018
Written by A. Petitet and R. Clint Whaley,  Innovative Computing Laboratory, UTK
Modified by Piotr Luszczek, Innovative Computing Laboratory, UTK
Modified by Julien Langou, University of Colorado Denver
================================================================================

An explanation of the input/output parameters follows:
T/V    : Wall time / encoded variant.
N      : The order of the coefficient matrix A.
NB     : The partitioning blocking factor.
P      : The number of process rows.
Q      : The number of process columns.
Time   : Time in seconds to solve the linear system.
Gflops : Rate of execution for solving the linear system.

The following parameter values will be used:

N      :   24576    24576
NB     :    2048
PMAP   : Row-major process mapping
P      :       1
Q      :       1
PFACT  :    Left
NBMIN  :       2
NDIV   :       2
RFACT  :    Left
BCAST  :   1ring
DEPTH  :       1
SWAP   : Spread-roll (long)
L1     : no-transposed form
U      : no-transposed form
EQUIL  : yes
ALIGN  : 8 double precision words

--------------------------------------------------------------------------------

- The matrix A is randomly generated for each test.
- The following scaled residual check will be computed:
      ||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N )
- The relative machine precision (eps) is taken to be               1.110223e-16
- Computational tests pass if scaled residuals are less than                16.0

================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       24576  2048     1     1             228.26             4.3357e+01
HPL_pdgesv() start time Fri Feb  2 14:50:51 2024

HPL_pdgesv() end time   Fri Feb  2 14:54:39 2024

--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   4.05852126e+08 ...... FAILED
||Ax-b||_oo  . . . . . . . . . . . . . . . . . =        5455.998041
||A||_oo . . . . . . . . . . . . . . . . . . . =        6229.846527
||A||_1  . . . . . . . . . . . . . . . . . . . =        6230.505776
||x||_oo . . . . . . . . . . . . . . . . . . . =         790.874477
||x||_1  . . . . . . . . . . . . . . . . . . . =      203978.961025
||b||_oo . . . . . . . . . . . . . . . . . . . =           0.499978
================================================================================
T/V                N    NB     P     Q               Time                 Gflops
--------------------------------------------------------------------------------
WR10L2L2       24576  2048     1     1             222.76             4.4426e+01
HPL_pdgesv() start time Fri Feb  2 14:54:57 2024

HPL_pdgesv() end time   Fri Feb  2 14:58:40 2024

--------------------------------------------------------------------------------
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   4.05852126e+08 ...... FAILED
||Ax-b||_oo  . . . . . . . . . . . . . . . . . =        5455.998041
||A||_oo . . . . . . . . . . . . . . . . . . . =        6229.846527
||A||_1  . . . . . . . . . . . . . . . . . . . =        6230.505776
||x||_oo . . . . . . . . . . . . . . . . . . . =         790.874477
||x||_1  . . . . . . . . . . . . . . . . . . . =      203978.961025
||b||_oo . . . . . . . . . . . . . . . . . . . =           0.499978
================================================================================

Finished      2 tests with the following results:
              0 tests completed and passed residual checks,
              2 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.

```
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the GNU General Public License version 2. See
[License.txt](https://github.com/oneapi-src/Velocity-Bench/blob/main/hplinpack/LICENSE.md) for details.
