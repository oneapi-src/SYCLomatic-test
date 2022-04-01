runfile: main.o ./inner/foo.o ./inner/bar.o 
	nvcc main.o ./inner/foo.o ./inner/bar.o -I/usr/local/cuda/include -o runfile

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc $(OUTPUT) $(KERNEL_DIM) main.cu -I./inner -c -O3

./inner/foo.o ./inner/avimod.o:
	cd inner; make;

# delete all object files
clean:
	rm -f *.o inner/*.o runfile
