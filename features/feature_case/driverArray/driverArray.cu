int main(){
    int *data;
    size_t width, height, depth, pitch, woffset, hoffset;
    CUstream cs;
    CUarray acu;
    cuMemcpyHtoA(acu, woffset, data, width);
    cuMemcpyAtoH(data, acu, woffset, width);
    cuMemcpyHtoAAsync(acu, woffset, data, width, cs);
    cuMemcpyAtoHAsync(data, acu, woffset, width, cs);

    CUdeviceptr data2;
    cuMemAlloc(&data2, sizeof(int) * 30);
    cuMemcpyDtoA(acu, woffset, data2, width);
    cuMemcpyAtoD(data2, acu, woffset, width);

    CUarray acu2;
    cuMemcpyAtoA(acu, woffset, acu2, woffset, width);
}