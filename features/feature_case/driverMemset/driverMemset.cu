
int main(){
    int size = 32;
    CUdeviceptr f_D = 0;
    cuMemAlloc(&f_D, size);
    CUstream stream;
    unsigned int v32 = 50000;
    unsigned short v16 = 20000;
    unsigned char v8 = (unsigned char) 200;
    cuMemsetD32(f_D, v32, size);
    cuMemsetD16(f_D, v16, size * 2);
    cuMemsetD8(f_D, v8, size * 4);
    cuMemsetD32Async(f_D, v32, size, stream);
    cuMemsetD16Async(f_D, v16, size * 2, stream);
    cuMemsetD8Async(f_D, v8, size * 4, stream);
    cuMemsetD2D32(f_D, 1, v32, 4, 6);
    cuMemsetD2D16(f_D, 1, v16, 4 * 2, 6);
    cuMemsetD2D8(f_D, 1, v8, 4 * 4, 6);
    cuMemsetD2D32Async(f_D, 1, v32, 4, 6, stream);
    cuMemsetD2D16Async(f_D, 1, v16, 4 * 2, 6, stream);
    cuMemsetD2D8Async(f_D, 1, v8, 4 * 4, 6, stream);
}