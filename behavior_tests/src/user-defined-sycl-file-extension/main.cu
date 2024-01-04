// we only care about file extension after migration so a minimul device code
__global__
void do_nothing_kernel() {}

int main(void) {
    do_nothing_kernel<<<1,1>>>();
    return 0;
}
