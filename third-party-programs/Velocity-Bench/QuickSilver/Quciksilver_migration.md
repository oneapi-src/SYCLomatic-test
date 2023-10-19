# SYCLomatic Tool: Migrate Quicksilver APP
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

1. This sample project contains a simple CUDA program with 111 files located in ```CUDA``` directory and the sub-directory src of ```CUDA```:

```
CUDA
└── src
    ├── AtomicMacro.hh
    ├── BulkStorage.hh
    ├── CollisionEvent.hh
    ├── CommObject.hh
    ├── CoralBenchmark.cc
    ├── CoralBenchmark.hh
    ├── CycleTracking.cc
    ├── CycleTracking.hh
    ├── DeclareMacro.hh
    ├── DecompositionObject.cc
    ├── DecompositionObject.hh
    ├── DirectionCosine.cc
    ├── DirectionCosine.hh
    ├── EnergySpectrum.cc
    ├── EnergySpectrum.hh
    ├── FacetPair.hh
    ├── GlobalFccGrid.cc
    ├── GlobalFccGrid.hh
    ├── Globals.hh
    ├── GridAssignmentObject.cc
    ├── GridAssignmentObject.hh
    ├── IndexToTuple.hh
    ├── IndexToTuple4.hh
    ├── InputBlock.cc
    ├── InputBlock.hh
    ├── Long64.hh
    ├── MCT.hh
    ├── MC_Base_Particle.cc
    ├── MC_Base_Particle.hh
    ├── MC_Cell_State.hh
    ├── MC_Distance_To_Facet.hh
    ├── MC_Domain.cc
    ├── MC_Domain.hh
    ├── MC_Facet_Adjacency.hh
    ├── MC_Facet_Crossing_Event.hh
    ├── MC_Facet_Geometry.hh
    ├── MC_Fast_Timer.cc
    ├── MC_Fast_Timer.hh
    ├── MC_Location.hh
    ├── MC_Nearest_Facet.hh
    ├── MC_Particle.hh
    ├── MC_Particle_Buffer.cc
    ├── MC_Particle_Buffer.hh
    ├── MC_Processor_Info.hh
    ├── MC_RNG_State.hh
    ├── MC_Segment_Outcome.hh
    ├── MC_SourceNow.cc
    ├── MC_SourceNow.hh
    ├── MC_Time_Info.hh
    ├── MC_Vector.hh
    ├── MacroscopicCrossSection.hh
    ├── Makefile
    ├── MaterialDatabase.hh
    ├── MemoryControl.hh
    ├── MeshPartition.cc
    ├── MeshPartition.hh
    ├── MonteCarlo.cc
    ├── MonteCarlo.hh
    ├── MpiCommObject.cc
    ├── MpiCommObject.hh
    ├── NVTX_Range.hh
    ├── NuclearData.hh
    ├── Parameters.cc
    ├── Parameters.hh
    ├── ParticleVault.cc
    ├── ParticleVault.hh
    ├── ParticleVaultContainer.cc
    ├── ParticleVaultContainer.hh
    ├── PhysicalConstants.cc
    ├── PhysicalConstants.hh
    ├── PopulationControl.cc
    ├── PopulationControl.hh
    ├── QS_Vector.hh
    ├── Random.cc
    ├── Random.h
    ├── SendQueue.cc
    ├── SendQueue.hh
    ├── SharedMemoryCommObject.cc
    ├── SharedMemoryCommObject.hh
    ├── Tallies.cc
    ├── Tallies.hh
    ├── Tuple.hh
    ├── Tuple4.hh
    ├── Tuple4ToIndex.hh
    ├── TupleToIndex.hh
    ├── cmdLineParser.cc
    ├── cmdLineParser.hh
    ├── cudaFunctions.cc
    ├── cudaFunctions.hh
    ├── cudaUtils.hh
    ├── git_hash.hh
    ├── git_vers.hh
    ├── initMC.cc
    ├── initMC.hh
    ├── macros.hh
    ├── main.cc
    ├── mc_omp_critical.hh
    ├── mc_omp_parallel_for_schedule_static.hh
    ├── mc_omp_parallel_for_schedule_static_if.hh
    ├── mc_omp_parallel_for_schedule_static_num_physical_cores.hh
    ├── memUtils.hh
    ├── mpi_stubs.hh
    ├── mpi_stubs_internal.hh
    ├── parseUtils.cc
    ├── parseUtils.hh
    ├── portability.hh
    ├── qs_assert.hh
    ├── utils.cc
    ├── utils.hh
    ├── utilsMpi.cc
    └── utilsMpi.hh
```
2. Use the **intercept-build** tool to intercept the ```makefile``` build steps to generate the compilation database `compile_commands.json` file under the same fodler.
``` sh
$ cd CUDA/src
$ intercept-build make
```
2. Use the tool's `--in-root` option and provide input files to specify where
   to locate the CUDA files that needs migration; use the tool’s `--out-root`
   option to designate where to generate the resulting files(default is `dpct_output`); use the tool's `-p` option to specify compilation database to migrate the whole project:

```sh
# From the CUDA/src directory as root directory:
$ dpct --in-root=. -p=./compile_commands.json --out-root=out --gen-build-script --cuda-include-path=/usr/local/cuda/include
```

> If an `--in-root` option is not specified, the directory of the first input
> source file is implied. If `--out-root` is not specified, `./dpct_output`
> is implied.

You should see the migrated files in the `out` folder that was specified
by the `--out-root` option.

3. Modify the ``-std=c++11`` to ``-std=c++17``, due to the ``icpx`` compiler needs to build with the at latest version c++17.

4. Inspect the migrated source code, address any `DPCT` warnings generated
   by the Intel® DPC++ Compatibility Tool, and verify the new program correctness.

Warnings are printed to the console and added as comments in the migrated
source. See *Diagnostic Reference* in the [Intel® DPC++ Compatibility Tool Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html) for more information on what each warning means.


This sample should generate the following warning:

```
warning:  /*
    DPCT1040:0: Use sycl::stream instead of printf if your code is used on the
    device.
    */
```

See below **Addressing Warnings in the Migrated Code** to understand how to
resolve the warning.


4. Build the migrated code with generated Makefile.dpct

```
$ cd out
$ make -f Makefile.dpct
# *Note:* Please make sure the oneAPI package was installed before building the application to make sure the oneAPI DPC++ compiler was installed.
```

# Addressing Warnings in Migrated Code

Migration generated one warning for code that `dpct` could not migrate:

```
warning:  /*
    DPCT1040:15: Use sycl::stream instead of printf if your code is used on the
    device.
    */
```

As you have noticed, the migration of this project resulted in one DPCT
message that needs to be addressed, **DPCT1040**. Manually adjusting is needed to generate the SYCL compliant code if needed. The ``` PrintParticle ``` is host and device function, so the print may need to adjust to kernel. 

```
HOST_DEVICE_CUDA
inline void MC_Particle::PrintParticle() {
.......
/*
DPCT1040:0: Use sycl::stream instead of printf if your code is used on the device.
*/
printf("coordiante:          %g\t%g\t%g\n", coordinate.x, coordinate.y,
      coordinate.z);
.......
}
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
$ ./qs -i ../../../Examples/AllScattering/scatteringOnly.inp
Copyright (c) 2016
Lawrence Livermore National Security, LLC
All Rights Reserved
Quicksilver Version     : 2023-Mar-22-18:05:07
Quicksilver Git Hash    : 5b22f83364f62d6d8ec383402793a2ed38a25276
MPI Version             : 3.0
Number of MPI ranks     : 1
Number of OpenMP Threads: 1
Number of OpenMP CPUs   : 1

Simulation:
   dt: 1e-08
   fMax: 0.1
   inputFile: ../../../Examples/AllScattering/scatteringOnly.inp
   energySpectrum:
   boundaryCondition: octant
   loadBalance: 1
   cycleTimers: 0
   debugThreads: 0
   lx: 100
   ly: 100
   lz: 100
   nParticles: 10000000
   batchSize: 0
   nBatches: 10
   nSteps: 10
   nx: 10
   ny: 10
   nz: 10
   seed: 1029384756
   xDom: 0
   yDom: 0
   zDom: 0
   eMax: 20
   eMin: 1e-09
   nGroups: 230
   lowWeightCutoff: 0.001
   bTally: 1
   fTally: 1
   cTally: 1
   coralBenchmark: 0
   crossSectionsOut:

Geometry:
   material: sourceMaterial
   shape: brick
   xMax: 100
   xMin: 0
   yMax: 100
   yMin: 0
   zMax: 100
   zMin: 0

Material:
   name: sourceMaterial
   mass: 1000
   nIsotopes: 10
   nReactions: 9
   sourceRate: 1e+10
   totalCrossSection: 0.1
   absorptionCrossSection: flat
   fissionCrossSection: flat
   scatteringCrossSection: flat
   absorptionCrossSectionRatio: 0
   fissionCrossSectionRatio: 0
   scatteringCrossSectionRatio: 1

CrossSection:
   name: flat
   A: 0
   B: 0
   C: 0
   D: 0
   E: 1
   nuBar: 2.4
Building partition 0
Building partition 1
Building partition 2
Building partition 3
Building MC_Domain 0
Building MC_Domain 1
Building MC_Domain 2
Building MC_Domain 3
Starting Consistency Check
Finished Consistency Check
Finished initMesh
   cycle  start  source     rr  split     absorb    scatter fission    produce    collisn   escape     census    num_seg   scalar_flux      cycleInit  cycleTracking  cycleFinalize
       0      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   2.607000e-03   1.297000e-03   1.000000e-06
       1      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   1.180000e-04   1.000000e-06   0.000000e+00
       2      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   5.600000e-05   1.000000e-06   0.000000e+00
       3      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   3.600000e-05   1.000000e-06   0.000000e+00
       4      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   3.600000e-05   1.000000e-06   0.000000e+00
       5      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   3.600000e-05   1.000000e-06   0.000000e+00
       6      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   3.600000e-05   1.000000e-06   0.000000e+00
       7      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   3.600000e-05   1.000000e-06   0.000000e+00
       8      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   4.100000e-05   1.000000e-06   0.000000e+00
       9      0       0      0      0          0          0       0          0          0        0          0          0  0.000000e+00   3.600000e-05   1.000000e-06   0.000000e+00

Timer                       Cumulative   Cumulative   Cumulative   Cumulative   Cumulative   Cumulative
Name                            number    microSecs    microSecs    microSecs    microSecs   Efficiency
                              of calls          min          avg          max       stddev       Rating
main                                 1    6.015e+03    6.015e+03    6.015e+03    0.000e+00       100.00
cycleInit                           10    3.038e+03    3.038e+03    3.038e+03    0.000e+00       100.00
cycleTracking                       10    1.306e+03    1.306e+03    1.306e+03    0.000e+00       100.00
cycleTracking_Kernel               100    0.000e+00    0.000e+00    0.000e+00    0.000e+00         0.00
cycleTracking_MPI                  110    1.294e+03    1.294e+03    1.294e+03    0.000e+00       100.00
cycleTracking_Test_Done              0    0.000e+00    0.000e+00    0.000e+00    0.000e+00         0.00
cycleFinalize                       20    2.790e+02    2.790e+02    2.790e+02    0.000e+00       100.00
Figure Of Merit                   0.00 [Num Mega Segments / Cycle Tracking Time]
```
**Note:** The testing result was running on Intel(R) Core(TM) i7-13700K on the CPU backend with 2023.2 oneAPI released oneAPI packaged. 

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html).

## License
Code samples are licensed under the GNU General Public License version 2. See
[License.txt](https://github.com/oneapi-src/Velocity-Bench/blob/main/Quicksilver/LICENSE.md) for details.
