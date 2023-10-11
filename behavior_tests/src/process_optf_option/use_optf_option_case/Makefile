all:test

test:test1.o test2.o

test1.o:test1.cu
	nvcc -c test1.cu --options-file=./includes1.rsp -optf=./includes2.rsp --options-file ./includes3.rsp -optf ./includes4.rsp

test2.o:test2.cu
	cd inner && nvcc -c ../test2.cu



clean:
	-rm inner/test2.o test1.o