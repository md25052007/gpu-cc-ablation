# Build:
#   make            -> bin/cc            (timing build, no instrumentation)
#   make counters   -> bin/cc_counters   (CAS / load-balance counters; never use for timing)
#   make ARCH=sm_75 -> pick the GPU architecture explicitly (T4 = sm_75, A100 = sm_80,
#                      RTX 30xx = sm_86, RTX 40xx = sm_89). Default: native (the GPU in this machine).
ARCH   ?= native
NVCC   ?= nvcc
FLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
SRC    := src/cc.cu
DEPS   := src/cc.cu src/kernels.cuh src/graph.h

all: bin/cc

counters: bin/cc_counters

bin/cc: $(DEPS)
	@mkdir -p bin
	$(NVCC) $(FLAGS) -o $@ $(SRC)

bin/cc_counters: $(DEPS)
	@mkdir -p bin
	$(NVCC) $(FLAGS) -DCC_COUNTERS -o $@ $(SRC)

# 12 small test graphs with awkward shapes; every version checked against the CPU reference.
tests: bin/cc
	python3 tools/run_tests.py

# Same checks without a GPU: kernel logic run sequentially on the CPU.
emutests: bin/cc
	python3 tools/run_tests.py --emulate

clean:
	rm -rf bin

.PHONY: all counters tests emutests clean
