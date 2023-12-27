#!/usr/bin/env bash
export SHELL=/bin/bash
testName=ct-1306
testDirname=test
subDirname=subdir
symDirname=symdir

srcDir=$(cd "$(dirname "$0")" ; pwd -P)
curDir=$(pwd -P)
testDir=$curDir/$testDirname

function cleanup() {
  if [ -d $testDir ]; then
    rm -rf $testDir
  fi
}

# create a regular dir in current dir named $testDirname
if [ -d $testDirname -o -f $testDirname -o -L $testDirname ]; then
  echo "ERROR: file or symlink named $testDirname already exists"
  exit
fi
mkdir $testDirname

# copy source files to test
cd $testDirname
cp $srcDir/$testName.cu $srcDir/$testName.h $srcDir/$testName.h.cmp .

# create a regular subdir
mkdir $subDirname
cd $subDirname

# create a symlink dir to point to parent dir
ln -s .. $symDirname
cd $symDirname

# try to migrate .cu file from symlink directory
dpct -in-root=. --enable-profiling=0 -out-root=out --cuda-include-path=$CUDA_INCLUDE_PATH ${testName}.cu

# check if .cu was migrated
if [ ! -f out/$testName.dp.cpp ]; then
  echo "FAIL: $testName.dp.cpp wasn't created"
  exit -1
fi

# check if .h was migrated
if [ ! -f out/$testName.h ]; then
  echo "FAIL: $testName.dp.cpp wasn't created"
  exit -1
fi

# check if migration warning was correctly generated
diff out/$testName.h $testName.h.cmp
if [ $? -ne 0 ]; then
  echo "FAIL: Expected migration warning wasn't created"
  exit -1
fi

cleanup

echo "PASSED"
exit
