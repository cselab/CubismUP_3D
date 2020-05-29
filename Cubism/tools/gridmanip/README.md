# Grid manipulation tool

The tool uses the I/O routines provided by the Cubism library to perform
various grid operations.  If necessary, additional I/O routines can be added
to support non-native data formats.

## Prolongation

Computes a refined grid using the interpolation scheme suggested by Harten et
al. (1995 and 1997), see also the file `Prolongation/MPI_GridTransfer.h`.
Refinement steps that are a power of two of the input grid are supported at the
moment.

## Restriction

Computes a coarsening of the provided input grid by averaging adjacent cells
into a coarser cell.  Coarsening by a power of two is supported at the moment.

## Smoothing

This operation performs a number of smoothing steps on based on the data of the
input grid.  A 7-point or 27-point stencil is supported at the moment.
