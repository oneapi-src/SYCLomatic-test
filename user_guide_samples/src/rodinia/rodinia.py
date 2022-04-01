# ====------ rodinia.py---------- *- Python -* ----===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
# ===----------------------------------------------------------------------===#
import subprocess
import os
import sys

p = "--cuda-include-path=" + os.environ["CUDA_INCLUDE_PATH"] + " " + sys.argv[1]

# prepare env
os.chdir("./rodinia_3.1/cuda/nw/")
# migrate code
subprocess.call(["intercept-build","/usr/bin/make"])
subprocess.call(("dpct -p compile_commands.json -in-root=" + os.getcwd() + " -out-root=dpcpp_out *.cu " + p), shell=True)
print("Migration done!!")
subprocess.call("ls")

# modify code
os.chdir("./dpcpp_out")
sourceFile = open("needle.dp.cpp","r")
lines = sourceFile.readlines()
sourceFile.close()
newLines = []
for line in lines:
    newLines.append(line.replace("17/*BLOCK_SIZE+1*/","BLOCK_SIZE+1").replace("16/*BLOCK_SIZE*/","BLOCK_SIZE"))

outFile = open("needle.dp.cpp","w")
for line in newLines:
    outFile.write(line)
outFile.close()

sourceFile = open("needle.h","r")
lines = sourceFile.readlines()
sourceFile.close()
newLines = []
for line in lines:
    newLines.append(line.replace("#define BLOCK_SIZE 16","#define BLOCK_SIZE 128"))

outFile = open("needle.h","w")
for line in newLines:
    outFile.write(line)
outFile.close()
print("Modification done!!")

# build code
makeFile = open("Makefile", "w")
makeFile.write("ifndef ($(CC))\n")
makeFile.write("\tCC=dpcpp\n")
makeFile.write("endif\n")
makeFile.write("CXXFLAGS= -Wno-error=sycl-strict \n")
makeFile.write("LDFLAGS= \n")
makeFile.write("INCFLAGS= -I${DPCT_BUNDLE_ROOT}/include\n")
makeFile.write("SOURCES = needle.dp.cpp\n")
makeFile.write("EXEC=nw\n")
makeFile.write("all:\n")
makeFile.write("\t$(CC) $(INCFLAGS) $(CXXFLAGS) $(SOURCES) -o $(EXEC) \n")
makeFile.write("clean:\n")
makeFile.write("\trm -rf nw\n")
makeFile.close()
subprocess.call("/usr/bin/make")
print("Build done!!")

# # check run result
if subprocess.check_output(["./nw","32","10"])==b'WG size of kernel = 128 \nStart Needleman-Wunsch\nProcessing top-left matrix\nProcessing bottom-right matrix\n':
    print("case pass")
else:
    print("need modify the user guide of dpct");
    print("case fail")
