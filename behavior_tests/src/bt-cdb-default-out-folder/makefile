SRC = hello.cu
OBJ = test.o
EXEC = nvcc
Run: $(SRC)
	$(EXEC) -c -m64 -o $(OBJ) -D__CUDACC__=1 $(SRC)
