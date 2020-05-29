// File       : ProlongHarten.h
// Created    : Tue Jul 11 2017 03:01:34 PM (+0200)
// Author     : Fabian Wermelinger
// Description: Prolongation based on Harten (MPI_GridTransfer.h)
// Copyright 2017 ETH Zurich. All Rights Reserved.
#ifndef PROLONGHARTEN_H_AFYYD4ZZ
#define PROLONGHARTEN_H_AFYYD4ZZ

#include "GridOperator.h"
#include "Prolongation/MPI_GridTransfer.h"

#include <cassert>
#include <vector>

using namespace cubism;

template <typename TGridIn, typename TGridOut, typename TBlockLab>
class ProlongHarten : public GridOperator<TGridIn, TGridOut, TBlockLab>
{
public:
    ProlongHarten(ArgumentParser &p)
        : GridOperator<TGridIn, TGridOut, TBlockLab>(p)
    {
    }

    ~ProlongHarten() = default;

    void operator()(const TGridIn &grid_in,
                    TGridOut &grid_out,
                    const bool verbose) override
    {
        // 0.) checks
        assert(TGridIn::BlockType::sizeX == TGridOut::BlockType::sizeX);
        assert(TGridIn::BlockType::sizeY == TGridOut::BlockType::sizeY);
        assert(TGridIn::BlockType::sizeZ == TGridOut::BlockType::sizeZ);
        const int NX_in = grid_in.getResidentBlocksPerDimension(0);
        const int NY_in = grid_in.getResidentBlocksPerDimension(1);
        const int NZ_in = grid_in.getResidentBlocksPerDimension(2);
        const int NX_out = grid_out.getResidentBlocksPerDimension(0);
        const int NY_out = grid_out.getResidentBlocksPerDimension(1);
        const int NZ_out = grid_out.getResidentBlocksPerDimension(2);
        assert(NX_in <= NX_out);
        assert(NY_in <= NY_out);
        assert(NZ_in <= NZ_out);

        const size_t smooth_iter = this->m_parser("smooth_iter").asInt(0);

        interpolate_from_coarse<TGridIn, TGridOut, TBlockLab>(grid_in,
                                                              grid_out,
                                                              NX_in,
                                                              NY_in,
                                                              NZ_in,
                                                              NX_out,
                                                              NY_out,
                                                              NZ_out,
                                                              smooth_iter,
                                                              verbose);
    }
};

#endif /* PROLONGHARTEN_H_AFYYD4ZZ */
