BASENAME=FlowPastTinySphere
NNODE=1

FACTORY='IF3D_Sphere L=0.5 xpos=0.4 xvel=0.1 bForcedInSimFrame=1 bFixFrameOfRef=1
'
# for accel and decel start and stop add accel=1 T=time_for_accel
# shift center to shed vortices immediately by ypos=0.250244140625 zpos=0.250244140625

OPTIONS=
OPTIONS+=" -bpdx 4 -bpdy 4 -bpdz 4"
OPTIONS+=" -dump2D 1 -dump3D 1"
OPTIONS+=" -nprocsx ${NNODE}"
OPTIONS+=" -CFL 0.1 -DLM 10"
OPTIONS+=" -length 0.5"
OPTIONS+=" -nu 0.000025"
OPTIONS+=" -tend 10  -tdump 0.05"
