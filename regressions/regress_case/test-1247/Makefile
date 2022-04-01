objects = hello.o
CUDA_FLAGS = -Wno-deprecated-gpu-targets -arch=sm_70 --expt-relaxed-constexpr -ftz=true -prec-div=false -prec-sqrt=false -lineinfo -g -O3 -Xcompiler -    Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field   -Xcompiler \" -g -O3 -Wall \"  -std=c++14


hello : $(objects)
	nvcc -o hello $(objects)

hello.o : hello.c
	nvcc $(CUDA_FLAGS) -c `echo ./`hello.c

.PHONY : clean
clean :
	-rm hello *.o compile_commands.json