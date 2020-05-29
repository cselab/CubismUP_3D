#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))

#FACTORY='IF3D_CarlingFish L=0.2 T=1.0 xpos=0.35 bFixToPlanar=1 bFixFrameOfRef=1 heightProfile=danio widthProfile=danio
FACTORY='CarlingFish L=0.25 T=1.0 xpos=0.35 bFixToPlanar=1 bFixFrameOfRef=1
'

OPTIONS=
OPTIONS+=" -bpdx 64 -bpdy 32 -bpdz 16"
#OPTIONS+=" -bpdx 64 -bpdy 16 -bpdz 16"
OPTIONS+=" -dump2D 0 -dump3D 0 -restart 0"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.01"
OPTIONS+=" -length 0.2"
OPTIONS+=" -nu 0.000015625"
OPTIONS+=" -lambda 1e5 -use-dlm 10"
OPTIONS+=" -tend 6 -tdump 0.00"
