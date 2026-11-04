CFLAGS += -O3 -std=gnu11 -Wall -Wextra -fPIC -Iinclude
SRC = src/minnet.c src/gemm.c
OBJ = $(SRC:src/%.c=build/%.o)

ifeq ($(shell uname),Darwin)
CFLAGS += -DMN_ACCELERATE
LDLIBS = -framework Accelerate
LIB = build/libminnet.dylib
SHARED = -dynamiclib
else
LDLIBS = -lm
LIB = build/libminnet.so
SHARED = -shared
ifdef OPENBLAS
CFLAGS += -DMN_CBLAS
LDLIBS += -lopenblas
endif
endif

all: $(LIB) build/libminnet.a

$(LIB): $(OBJ)
	$(CC) $(SHARED) -o $@ $^ $(LDLIBS)

build/libminnet.a: $(OBJ)
	ar rcs $@ $^

build/%.o: src/%.c include/minnet.h src/gemm.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
