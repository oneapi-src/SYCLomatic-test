# Table of contents

[Overview](#Overview)

[Prerequisites](#Prerequisites)

[Contributing](#Contributing)

[Run tests](#run-tests)

[Test configuration file and test driver](#Test-configuration-file-and-test-driver)

[Add or modify test case](#Add-or-modify-test-case)


# Overview

This repo contains tests for SYCLomatic project.

# Prerequisites

   - SYCLomatic or Intel(R) DPC++ Compatibility Tool:
       - Build SYCLomatic from source [instructions](https://github.com/oneapi-src/SYCLomatic).
       - Or, download Intel(R) DPC++ Compatibility Tool [Binary](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#compatibility).
       - Or, install Intel(R) DPC++ Compatibility Tool from [oneAPI package](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
   - DPC++ compiler and oneAPI libraries:
       - Install from [oneAPI package](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
       - Or, build from source code:
           - DPC++ compiler [instructions](https://github.com/intel/llvm/blob/sycl/README.md)
                   or taken prebuilt from [releases](https://github.com/intel/llvm/releases).
           - [oneMKL library](https://github.com/oneapi-src/oneMKL/blob/develop/README.md).
           - [oneDPL library](https://github.com/oneapi-src/oneDPL/blob/main/README.md).
   - Runtime/Driver for GPU Device
       -  Target runtime(s) to execute tests on GPU devices. See [installation instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#install-low-level-runtime).
   - Python3.7
   - Certain CUDA\* header files may need to be accessible to the tool

# Contributing
See [CONTRIBUTING.md](https://github.com/oneapi-src/SYCLomatic-test/blob/SYCLomatic/CONTRIBUTING.md) for details.

# Run tests

All tests can be run by run_test.py python file.
```
    git clone https://github.com/oneapi-src/SYCLomatic-test.git
    cd SYCLomatic-test
    python3 run_test.py
```

To run the specific test suite. E.g: run the features test suite with the default option on the CPU device.
```
python3 run_test.py --suite features --option option_cpu
```
To run the specific test case. E.g: run the thrust-vector in the features test suite with --usm-level=none option on the CPU device.
```
python3 run_test.py --suite features --case thrust-vector --option option_usmnone_cpu
```

# Test configuration file and test driver
There are 3 levels configuration files:
   - Test suite list configuration file.
   - Test suite configuration file.
   - Test case configuration file.
For each test suite, there is a separate test driver.

## Test suite list configuration file: test_suite_list.xml

The configuration file test_suite_list.xml lists all the test suites target to be run. Each test suite is defined by "name", "dir" and "opts".

| Element Name | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| dir          | The relative path of the test suite.                         |
| name         | Name of test suite, also **${dir}/${name}.xml** is configuration file for this suite.           |
| opts         | Specifies the test option list for each test suite. Test options can be: **option_cpu**, **option_gpu**, **option_usmnone_cpu**, **option_cuda_backend** and so on. Each test option defines the target device to run and extra c2s options during migration, more details in the [Define option for a test suite](#define-option-for-a-test-suite-option_mappingjson). |

### Current test suite in SYCLomatic-test
| Test suite name | Description| Link with the CI|
| ------------ | ------------------------------------------------------------ |---|
|api_coverage | The test suite includes API migration tests cases. Also reference to https://github.com/oneapi-src/SYCLomatic/tree/SYCLomatic/clang/test/dpct for more API migration test cases. |test-api_coverage|
|behavior_tests | The test suite includes both positive and negative behavior tests, target to validate the expected behavior of the SYCLomatic tool. |test-behavior_tests|
|help_function | The test suite contains the helper function test cases. New test case should be added if you implement a new helper function. The helper function is in https://github.com/oneapi-src/SYCLomatic/tree/SYCLomatic/clang/runtime/dpct-rt/include | test-help_function|
|features | The current test suite contains test cases for features of SYCLomatic. If a new feature is implemented, like enabling migration for new API(s), adding additional tool functionality, please make sure to include the corresponding end-to-end test cases in the suite. |test-features|
|regressions| The suite includes regression test cases which are captured during regular test. |test-regressions|
|samples|The suite consists of small end-to-end samples that cover various functionalities and scenarios.|test-samples|
|user_guide_samples| The suite includes test cases derived from the SYCLomatic user guide. |test-user_guide_samples|

### Define option for a test suite: option_mapping.json

The option_mapping.json defines extra options when run each test option.
The format of the option map is key and value pairs. The key is the test option name in format option_{.*}_{cpu|gpu}, cpu and gpu here define the target device on which this test option will run. The value defines the specific options. Following are examples of option map:

| Option map (key:value pair)            | Option map Description                                                     |
| -------------------------------------- | ------------------------------------------------------------ |
| "option_cpu" : ""                         | a. c2s migrates with default opt. b. The test run on the CPU device. |
| "option_usmnone_cpu" : "--usm-level=none" | a. c2s migrates with usm switching off.  b. The test run on the CPU device. |
| "option_gpu" : ""                         | a. c2s migrates with default opt. b. The test run on the GPU device. |
| "option_usmnone_gpu" : "--usm-level=none" | a. c2s migrates with usm switching off. b. The test run on the GPU device.  |
| "option_cuda_backend" : ""                      | a. c2s migrates with default opt. b. The test run on the CUDA backend. |


## Test suite configuration file:
Test Suite configuration file defines all the test cases in the same test suite, each test case is specified by a **testName** and **configFile**.
Following table gives more details on testName and configFile:
| Items to define a test case | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| testName                    | name for the test case.        |
|configFile    | configuration file for corresponding test case given by testName. See details of content of configFile in the [test case config file](#Test-case-configuration-file). |


## Test case configuration file
Test case configuration file defines the test case file and some rules to control whether the test case will be run or not.
As follow table give more details on key items: file and rule:
| Key item of test case configuration file. | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| \<file\>       | File Path of the test case.       |
| \<rule\>       | Rule for test case,  eg. A rule can be used to skip test when the case is not supported on the specific device or options. |
## Test driver for test suite
There is a test driver for each test suite, all test cases share the same test driver in a test suite.

There are two files for each test driver: **configuration file** and **implementation script file**.

For example, for test suite: internal_samples, there is a test driver configure file: "test_drivers.xml" and a test driver implementation file: test_samples.py. test_drivers.xml specifies the test driver implementation file(test_samples.py) in this test suite.

The test driver implementation file test_samples.py needs to implement the following 4 interfaces:

    1. setup_test():  Setup the execution environment. eg. setup CPATH or LD_LIBRARY_PATH in Linux to contain library required for the test case.
    2. migrate_test(): Migration command for each test case.
    3. build_test(): Compile and link command for each test case.
    4. run_test(): Run the test cases in the test suite.


# Add or modify test case

## Add a new test case
1. Prepare the test case: prepare the test case name(testName), test case configuration file(configFile) which defines the test case file(<file>) and rule to run the test case(rule), and the test case file itself.

2. Identify the target test suite to contain this test case and then update the test suite configuration file by adding new test case.

   In the test_suite_list.xml, for each entry there is a test suite configuration file (define as ${dir}/${name}.xml), and in test suite file (${dir}/${name}.xml), add new test case defined by: testName and configFile.

3. Put the test case file to the right folder according to file value of test case configuration file.

## Update a test case
Identify what needs to be updated:

   - Update the functionality of test case:  modify the test file directly, the test file is defined in <file> field of test case configuration file.
   - Update test case on migration command, build command, run command: modify the corresponding code in test suite driver implementation file.
   - Update the test case run rule, like excluding test run on CPU device or Microsoft Windows platform: edit the <rule> field of test case configuration file.

## License

See [LICENSE](LICENSE) for details.

## Trademarks information
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.<br>
\*Other names and brands may be claimed as the property of others. SYCL is a trademark of the Khronos Group Inc.
