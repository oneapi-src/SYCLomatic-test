objects = hello.o

hello : $(objects)
	nvcc -o hello $(objects)

hello.o : hello.cu
	nvcc -c hello.cu

.PHONY : clean
clean :
	-rm hello hello.o compile_commands.json

