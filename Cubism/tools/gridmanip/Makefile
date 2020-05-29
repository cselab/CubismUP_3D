SHELL := /bin/bash

CC = mpic++

bs ?= 32
align ?= 16
omp ?= 1
ap ?= float
config ?= release

# HDF5 is required to build
hdf = 1
hdf-inc ?= ${HDF5_ROOT}/include
hdf-lib ?= ${HDF5_ROOT}/lib

GIT_COMMIT := -D_GIT_HASH_=\"$(shell git rev-parse --verify HEAD)\"
CPPFLAGS += $(GIT_COMMIT) -g -std=c++11 -D_HARTEN5_
CPPFLAGS += $(extra)

# compiler warnings
CPPFLAGS += -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS += -Wwrite-strings -Wmissing-declarations -Wredundant-decls
CPPFLAGS += -Wshadow -Woverloaded-virtual -Wuninitialized
CPPFLAGS += -Wpedantic -Wno-unused-parameter

ifeq "$(omp)" "1"
	CPPFLAGS += -fopenmp
	OPTFLAGS += -fopenmp
endif

ifeq "$(ap)" "float"
	CPPFLAGS += -D_FLOAT_PRECISION_
endif

ifeq "$(config)" "release"
	OPTFLAGS += -O3 -DNDEBUG
endif

ifeq "$(hdf)" "1"
	CPPFLAGS += -I$(hdf-inc) -DCUBISM_USE_HDF
	LIBS += -L$(hdf-lib) -lhdf5
endif

CPPFLAGS += -DCUBISM_ALIGNMENT=$(align) -D_BLOCKSIZE_=$(bs) -D_BLOCKSIZEX_=$(bs) -D_BLOCKSIZEY_=$(bs) -D_BLOCKSIZEZ_=$(bs)
CPPFLAGS += -I.
CPPFLAGS += -I../../../Cubism/include

.DEFAULT_GOAL := gridmanip

OBJECTS = main.o \
		  ../../../Cubism/src/ArgumentParser.o

all: gridmanip

gridmanip: $(OBJECTS)
	$(CC) $(OPTFLAGS) $(extra) $^ -o $@ $(LIBS)

%.o: %.cpp
	$(CC)  $(OPTFLAGS) $(CPPFLAGS) -c $^ -o $@

clean:
	rm -f *.o gridmanip *~
