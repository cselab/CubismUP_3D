#!/bin/bash
BASENAME=tiny_carling_Re2k
NNODEX=1
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))

FACTORY='IF3D_CarlingFish L=0.8 T=1.0 xpos=0.45 bFixToPlanar=1 bFixFrameOfRef=1 bForcedInSimFrame=1 heightProfile=danio widthProfile=danio
'

OPTIONS=
OPTIONS+=" -bpdx 8 -bpdy 1 -bpdz 1"
OPTIONS+=" -restart 0"
OPTIONS+=" -nprocsx ${NNODEX}"
OPTIONS+=" -nprocsy ${NNODEY}"
OPTIONS+=" -nprocsz 1"
OPTIONS+=" -CFL 0.1"
OPTIONS+=" -length 0.8"
OPTIONS+=" -nu 0.00032"
OPTIONS+=" -tend 5 -tdump 0.05"
