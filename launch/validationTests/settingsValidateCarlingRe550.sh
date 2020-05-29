#!/bin/bash

BASENAME=validateCarling550_try2_lmax7
NNODE=32
NNODEX=32
NNODEY=1
WCLOCK=24:00:00
WSECS=43000
FFACTORY=factoryCarling

OPTIONS=
#OPTIONS+=" -bpdx 64 -bpdy 64 -bpdz 64"
OPTIONS+=" -bpdx 128 -bpdy 64 -bpdz 64"
OPTIONS+=" -2Ddump 0 -restart 0"
OPTIONS+=" -nprocsx ${NNODEX}"
OPTIONS+=" -nprocsy ${NNODEY}"
OPTIONS+=" -nprocsz 1"
OPTIONS+=" -CFL 0.1"
OPTIONS+=" -length 0.25"
OPTIONS+=" -lambda 1e4"
#OPTIONS+=" -lambda 1e6"
OPTIONS+=" -nu 0.0001136363636"
OPTIONS+=" -tend 8 -tdump 0.2"
