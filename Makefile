# granite.cpp — Granite Speech 5.0 pure C CTC inference
#
# The model is encoder-only, so there is a single build shape: CPU with BLAS.

CC = clang
CFLAGS_BASE = -Wall -Wextra -O3 -march=native
LDFLAGS = -lm -lpthread

UNAME_S := $(shell uname -s)

SRCS = granite.c granite_kernels.c granite_audio.c granite_encoder.c \
       granite_tokenizer.c granite_safetensors.c
OBJS = $(SRCS:.c=.o)
TARGET = granite
HEADERS = granite.h granite_kernels.h granite_audio.h granite_tokenizer.h \
          granite_safetensors.h

.PHONY: all blas debug clean test help

all: help

help:
	@echo "granite.cpp — Granite Speech 5.0 pure C inference"
	@echo ""
	@echo "  make blas    - build with BLAS (Accelerate on macOS, OpenBLAS on Linux)"
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

debug: CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Layer-by-layer comparison against tensors dumped by reference.py
test:
	./test_granite.py

clean:
	rm -f $(OBJS) main.o $(TARGET)
