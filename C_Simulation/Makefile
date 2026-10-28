# Run from Git Bash or the MSYS2 UCRT64 terminal: mingw32-make (or make)

CC      = gcc
CFLAGS  = -O2 -std=c11 -Wall -Wextra -Wpedantic -Iinclude
LDLIBS  = -lm

SRC = src/pla_exp.c src/softmax_approx.c

ifeq ($(OS),Windows_NT)
  LIB = build/dlha_approx.dll
  EXE = .exe
  LDFLAGS_LIB = -shared -static-libgcc
else
  LIB = build/libdlha_approx.so
  EXE =
  CFLAGS += -fPIC
  LDFLAGS_LIB = -shared
endif

TEST = build/test_softmax_approx$(EXE)

.PHONY: all test clean

all: $(LIB) $(TEST)

$(LIB): $(SRC) | build
	$(CC) $(CFLAGS) -DDLHA_BUILD_DLL $(LDFLAGS_LIB) -o $@ $(SRC) $(LDLIBS)

$(TEST): tests/test_softmax_approx.c $(SRC) | build
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test: $(TEST)
	$(TEST)

build:
	mkdir -p build

clean:
	rm -rf build
