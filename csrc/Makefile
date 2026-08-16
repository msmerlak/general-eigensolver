# Build the C purification eigensolver.
#
# It links the SAME OpenBLAS that NumPy uses (scipy_openblas64, 64-bit int,
# scipy_*_64_ symbol prefix). That is deliberate: benchmarking the C solver
# against a differently-built BLAS would repeat SSJ_LOG #13's confound, where
# SciPy's BLAS wrapper measured 2.6x slower than NumPy's `@` on the same box
# and nearly produced a false "syrk is useless" verdict.
#
#   make            build ./bench
#   make run        build and run the accuracy battery + wall race
#   make BLAS=/path/to/libopenblas.so   override the BLAS

BLAS ?= $(shell python3 -c "import glob,sys; \
        g=glob.glob('/usr/local/lib/python3.11/dist-packages/numpy.libs/libscipy_openblas64*.so'); \
        print(g[0] if g else '')")

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fno-math-errno -Wall -Wextra
LDFLAGS ?= -lm -Wl,-rpath,$(dir $(BLAS))

bench: purify_eigh.c bench.c
	@test -n "$(BLAS)" || { echo "no BLAS found; set BLAS=/path/to/lib"; exit 1; }
	$(CC) $(CFLAGS) -o $@ purify_eigh.c bench.c $(BLAS) $(LDFLAGS)

sdc_bench: sdc_eig.c sdc_bench.c sdc_eig.h
	@test -n "$(BLAS)" || { echo "no BLAS found; set BLAS=/path/to/lib"; exit 1; }
	$(CC) $(CFLAGS) -o $@ sdc_eig.c sdc_bench.c $(BLAS) $(LDFLAGS)

all: bench sdc_bench

run: bench
	./bench 5

run-sdc: sdc_bench
	./sdc_bench 3

clean:
	rm -f bench sdc_bench

.PHONY: all run run-sdc clean
