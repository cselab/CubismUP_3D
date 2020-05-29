# File       : Makefile
# Created    : Wed Nov 07 2018 06:53:55 PM (+0100)
# Author     : Fabian Wermelinger
# Description: Run test cases.  You need to compile the gridmanip tool and
#              create a copy or sym link in the directory your run this
#              makefile.
# Copyright 2018 ETH Zurich. All Rights Reserved.

.PHONY: all clean

all:
	./gridmanip -conf smoother.conf
	./gridmanip -conf restrict.conf
	./gridmanip -conf prolong.conf

clean:
	rm -f *.h5 *.xmf
