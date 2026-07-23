all:gemm-test

CC=gcc
#CC=icx

ifeq ($(CC),gcc)
	CFLAGS= -fopenmp -O3 -g -march=native
	LDFLAGS=-lgomp
else
	CFLAGS= -qopenmp -O3 -g  -march=native 
	LDFLAGS=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm
endif

gemm-test: gemm-test.o
	$(CC) -o $@ $< $(LDFLAGS)

run:
	./gemm-test 960 10



%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o gemm-test

