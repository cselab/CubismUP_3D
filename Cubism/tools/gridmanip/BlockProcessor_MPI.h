// File       : BlockProcessor_MPI.h
// Created    : Sat May 04 2019 01:25:51 PM (+0200)
// Author     : Fabian Wermelinger
// Description: Simple MPI block-processor
// Copyright 2019 ETH Zurich. All Rights Reserved.
#ifndef BLOCKPROCESSOR_MPI_H_VTNAQSNJ
#define BLOCKPROCESSOR_MPI_H_VTNAQSNJ

#include "Cubism/SynchronizerMPI.h"

#include <mpi.h>
#include <vector>

using namespace cubism;

template <typename TLab, typename TKernel, typename TGrid>
inline void
process(TKernel rhs, TGrid &grid, const Real t = 0.0, const bool record = false)
{
    TKernel myrhs = rhs;

    SynchronizerMPI<Real> &Synch = grid.sync(myrhs);

    std::vector<BlockInfo> avail0, avail1;

    const int nthreads = omp_get_max_threads();
    TLab *labs = new TLab[nthreads];
    for (int i = 0; i < nthreads; ++i)
        labs[i].prepare(grid, Synch);

    MPI_Barrier(grid.getCartComm());

    avail0 = Synch.avail_inner();
    BlockInfo *ary0 = &avail0.front();
#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        TLab &mylab = labs[tid];

#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < avail0.size(); i++) {
            mylab.load(ary0[i], t);
            rhs(mylab, ary0[i], *(typename TGrid::BlockType *)ary0[i].ptrBlock);
        }
    }

    avail1 = Synch.avail_halo();
    BlockInfo *ary1 = &avail1.front();
#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        TLab &mylab = labs[tid];

#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < avail1.size(); i++) {
            mylab.load(ary1[i], t);
            rhs(mylab, ary1[i], *(typename TGrid::BlockType *)ary1[i].ptrBlock);
        }
    }

    if (labs != NULL) {
        delete[] labs;
        labs = NULL;
    }

    MPI_Barrier(grid.getCartComm());
}

#endif /* BLOCKPROCESSOR_MPI_H_VTNAQSNJ */
