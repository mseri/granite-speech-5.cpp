# granite.cpp — Granite Speech 5.0 pure C CTC inference
#
# Two build shapes: CPU with BLAS, and Metal/MPS on any Mac with a Metal GPU
# (Apple Silicon or Intel). The MPS build is a superset — it keeps BLAS for
# the shapes too small to be worth a dispatch.

CC = clang
CFLAGS_BASE = -Wall -Wextra -O3 -march=native
LDFLAGS = -lm -lpthread

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

SRCS = granite.c granite_kernels.c granite_audio.c granite_encoder.c \
       granite_tokenizer.c granite_safetensors.c
OBJS = $(SRCS:.c=.o)
TARGET = granite
HEADERS = granite.h granite_kernels.h granite_audio.h granite_tokenizer.h \
          granite_safetensors.h

.PHONY: all blas mps debug clean test help

all: help

help:
	@echo "granite.cpp — Granite Speech 5.0 pure C inference"
	@echo ""
	@echo "  make blas    - build with BLAS (Accelerate on macOS, OpenBLAS on Linux)"
	@echo "  make mps     - build with the Metal/MPS backend (any Mac with Metal)"
	@echo "  make debug   - debug build with AddressSanitizer"
	@echo "  make test    - run the regression suite against reference.py"
	@echo "  make clean   - remove build artifacts"
	@echo ""
	@echo "Example: make blas && ./granite -d granite-speech-5.0 -i audio.wav"

ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# ---------------------------------------------------------------------- mps ---
# Objective-C sources need -fobjc-arc; the registry in granite_kernels_metal.m
# relies on ARC for its strong MTLBuffer references.
ifeq ($(UNAME_S),Darwin)
mps: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_MPS -DACCELERATE_NEW_LAPACK
mps: LDFLAGS += -framework Accelerate -framework Metal \
                -framework MetalPerformanceShaders -framework Foundation
mps: EXTRA_OBJS = granite_kernels_metal.o
mps:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" \
	         EXTRA_OBJS="$(EXTRA_OBJS)"
	@echo ""
	@echo "Built with the Metal/MPS backend"
else
mps:
	@echo "Error: 'make mps' requires macOS" >&2
	@echo "  this machine is $(UNAME_S)/$(UNAME_M); use 'make blas'" >&2
	@exit 1
endif

debug: CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

$(TARGET): $(OBJS) main.o $(EXTRA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

granite_kernels_metal.o: granite_kernels_metal.m granite_kernels_metal.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# Layer-by-layer comparison against tensors dumped by reference.py
test:
	./test_granite.py

clean:
	rm -f $(OBJS) main.o granite_kernels_metal.o $(TARGET)
