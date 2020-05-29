// File       : Smoother.h
// Created    : Sun Oct 29 2017 12:36:21 PM (+0100)
// Author     : Fabian Wermelinger
// Description: Smooth data
// Copyright 2017 ETH Zurich. All Rights Reserved.
#ifndef SMOOTHER_H_2PBGUG6D
#define SMOOTHER_H_2PBGUG6D

#include "Cubism/BlockInfo.h"
#include "GridOperator.h"
#include "Prolongation/MPI_GridTransfer.h"

#include <cassert>
#include <vector>

using namespace cubism;

template <typename TGridIn, typename TGridOut, typename TBlockLab>
class Smoother : public GridOperator<TGridIn, TGridOut, TBlockLab>
{
public:
    Smoother(ArgumentParser &p) : GridOperator<TGridIn, TGridOut, TBlockLab>(p)
    {
    }

    ~Smoother() = default;

    void operator()(const TGridIn &grid_in,
                    TGridOut &grid_out,
                    const bool verbose) override
    {
        // 0.) checks
        typedef typename TGridIn::BlockType TBlockIn;
        typedef typename TGridOut::BlockType TBlockOut;
        assert(TBlockIn::sizeX == TBlockOut::sizeX);
        assert(TBlockIn::sizeY == TBlockOut::sizeY);
        assert(TBlockIn::sizeZ == TBlockOut::sizeZ);
        assert(grid_in.getResidentBlocksPerDimension(0) ==
               grid_out.getResidentBlocksPerDimension(0));
        assert(grid_in.getResidentBlocksPerDimension(1) ==
               grid_out.getResidentBlocksPerDimension(1));
        assert(grid_in.getResidentBlocksPerDimension(2) ==
               grid_out.getResidentBlocksPerDimension(2));

        const size_t smooth_iter = this->m_parser("smooth_iter").asInt(0);

        // copy over
        std::vector<BlockInfo> info_in = grid_in.getResidentBlocksInfo();
        std::vector<BlockInfo> info_out = grid_out.getResidentBlocksInfo();
        assert(info_in.size() == info_out.size());

#pragma omp parallel for
        for (size_t i = 0; i < info_out.size(); i++) {
            BlockInfo infoout = info_out[i];
            TBlockOut &bout = *(TBlockOut *)infoout.ptrBlock;
            bout.clear(); // zero data
        }

#pragma omp parallel for
        for (size_t i = 0; i < info_in.size(); i++) {
            // src
            BlockInfo infoin = info_in[i];
            TBlockIn &bin = *(TBlockIn *)infoin.ptrBlock;

            // dst
            BlockInfo infoout = info_out[i];
            TBlockOut &bout = *(TBlockOut *)infoout.ptrBlock;

            for (int iz = 0; iz < TBlockIn::sizeZ; iz++)
                for (int iy = 0; iy < TBlockIn::sizeY; iy++)
                    for (int ix = 0; ix < TBlockIn::sizeX; ix++)
                        bout(ix, iy, iz) = bin(ix, iy, iz);
        }

        // smooth out grid
        for (size_t i = 0; i < smooth_iter; ++i) {
            if (verbose)
                std::cout << "smoothing grid: iteration " << i + 1 << std::endl;
            grid_smoother smoother;
            process<TBlockLab>(smoother, grid_out, 0, 0);
        }
    }
};

#endif /* SMOOTHER_H_2PBGUG6D */
