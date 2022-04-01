Before the patch https://git-amr-2.devtools.intel.com/gerrit/#/c/236198/ is applied, intercept-build will create database as below, and \"\\\"\" in it.

[
    {
        "command": "nvcc -c --cuda-gpu-arch=sm_70 -O3 - Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field \"\\\"\" -O3 -Wall \"\\\"\" -std=c++14 -D__CUDACC__=1 ./hello.c",
        "directory": "/home/cwsun/disk/FixBug/CTST-1247/intercept_ctst1247_test",
        "file": "/home/cwsun/disk/FixBug/CTST-1247/intercept_ctst1247_test/hello.c"
    }
]





After the patch is applied, \"\\\"\" will be removed in the generated compliation database below:
[
    {
        "command": "nvcc -c --cuda-gpu-arch=sm_70 -O3 Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field -O3 -Wall -std=c++14 -D__CUDACC__=1 ./hello.c",
        "directory": "/home/cwsun/disk/FixBug/CTST-1247/intercept_ctst1247_test",
        "file": "/home/cwsun/disk/FixBug/CTST-1247/intercept_ctst1247_test/hello.c"
    }
]