if [ -d out ]
then
rm -rf ./out
fi
dpct -out-root out --vcxprojfile=proj_c.vcxproj c_kernel.cu CuTmp_1.cu -cuda-include-path=${CUDA_INCLUDE_PATH} > output
grep "<None> node in proj_c.vcxproj and skipped" ./output
if [ "x0" = "x$?" ]
then
exit 0
fi
exit -1
