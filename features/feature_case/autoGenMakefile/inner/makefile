all: bar.o foo.o

bar.o: foo.h bar.c 
	gcc -c bar.c

foo.o: foo.h foo.c
	gcc -c foo.c

clean:
	rm *.o
