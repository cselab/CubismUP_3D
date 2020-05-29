SHELL := /bin/bash

CC = g++
extra ?=

CFLAGS = -g -O2 -I../../include
CFLAGS+= $(extra)
CFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align
CFLAGS+= -Wwrite-strings -Wmissing-declarations -Wredundant-decls
CFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized
CFLAGS+= -Wpedantic -Wno-unused-parameter # does not make much sense with function overloading

.PHONY: all clean

all: testArgumentParser.cpp
	$(CC) $(CFLAGS) -o testArgumentParser testArgumentParser.cpp ../../src/ArgumentParser.cpp

clean:
	rm -f testArgumentParser *.o
