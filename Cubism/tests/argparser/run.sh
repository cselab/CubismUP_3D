#!/usr/bin/env bash
# File       : run.sh
# Created    : Thu Mar 28 2019 06:11:33 PM (+0100)
# Author     : Fabian Wermelinger
# Description: Run testArgumentParser
# Copyright 2019 ETH Zurich. All Rights Reserved.
set -e

./testArgumentParser -conf test.conf \
    -arg1  1.0e-001 \
    -arg2 -1.0e-001 \
    -arg3  1.0e1 \
    -arg4 -1.0e+001 \
    -arg5 -1.0e400 \
    -arg6 +1.0e400 \
    -arg7  1.0e400 \
    -arg8  nan \
    -arg9  -nan \
    -arg10
