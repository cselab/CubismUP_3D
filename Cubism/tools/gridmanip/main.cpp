// File       : main.cpp
// Created    : Mon Jul 10 2017 12:18:47 PM (+0200)
// Author     : Fabian Wermelinger
// Description: Tool for restriction / prolongation of grid
// Copyright 2017 ETH Zurich. All Rights Reserved.
#ifndef CUBISM_USE_HDF
#pragma error "HDF5 library is required for build"
#endif /* CUBISM_USE_HDF */

#include "Cubism/ArgumentParser.h"
#include "Cubism/BlockInfo.h"
#include "Cubism/BlockLab.h"
#include "Cubism/BlockLabMPI.h"
#include "Cubism/Grid.h"
#include "Cubism/GridMPI.h"
#include "Cubism/HDF5Dumper_MPI.h"

// grid operators
#include "GridOperator.h"
#include "Prolongation/ProlongHarten.h"
#include "Restriction/RestrictBlockAverage.h"
#include "Smoother/Smoother.h"

// types required to process grids
#include "Types.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>

using namespace cubism;

int main(int argc, char *argv[])
{
    // Boundary conditions for prolongation (requires tensorial stencil) and
    // smoother (conventional 7-point stencil):
    //
    // periodic (naturally tensorial)
    // using LabMPI = BlockLabMPI<BlockLab<GridBlock>>;
    //
    // zeroth-order extrapolating boundary (copy first interior cell into halos)
    using LabMPI = BlockLabMPI<ExtrapolatingBoundaryTensorial<GridBlock>>;

    int provided;
    MPI_Init_thread(&argc, (char ***)&argv, MPI_THREAD_MULTIPLE, &provided);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool isroot = (0 == rank) ? true : false;

    ArgumentParser parser(argc, argv);
    if (parser.exist("conf"))
        parser.readFile(parser("conf").asString());

    // setup grids
    const Real extent = parser("extent").asDouble(1.0);
    const int xpesize = parser("xpesize").asInt(1);
    const int ypesize = parser("ypesize").asInt(1);
    const int zpesize = parser("zpesize").asInt(1);

    // fine blocks
    const int in_bpdx = parser("in_bpdx").asInt(1);
    const int in_bpdy = parser("in_bpdy").asInt(1);
    const int in_bpdz = parser("in_bpdz").asInt(1);

    // coarse blocks
    const int out_bpdx = parser("out_bpdx").asInt(1);
    const int out_bpdy = parser("out_bpdy").asInt(1);
    const int out_bpdz = parser("out_bpdz").asInt(1);

    using TGridIn = GridMPI<Grid<GridBlock, std::allocator>>;
    using TGridOut = GridMPI<Grid<GridBlock, std::allocator>>;

    TGridIn *const grid_in = new TGridIn(xpesize,
                                         ypesize,
                                         zpesize,
                                         in_bpdx,
                                         in_bpdy,
                                         in_bpdz,
                                         extent,
                                         MPI_COMM_WORLD);
    TGridOut *const grid_out = new TGridOut(xpesize,
                                            ypesize,
                                            zpesize,
                                            out_bpdx,
                                            out_bpdy,
                                            out_bpdz,
                                            extent,
                                            MPI_COMM_WORLD);

    // setup I/O
    InputOutputFactory<TGridIn, IOType::In> infactory(isroot, parser);
    InputOutputFactory<TGridOut, IOType::Out> outfactory(isroot, parser);

    // get data for input grid
    parser.set_strict_mode();
    const std::string infile = parser("in_file").asString();
    parser.unset_strict_mode();
    infactory(*grid_in, infile);

    // perform grid manipulation
    GridOperator<TGridIn, TGridOut, LabMPI> *gridmanip;
    const std::string operator_name =
        parser("operator").asString("RestrictBlockAverage");
    if (operator_name == "RestrictBlockAverage")
        gridmanip = new RestrictBlockAverage<TGridIn, TGridOut, LabMPI>(parser);
    else if (operator_name == "ProlongHarten")
        gridmanip = new ProlongHarten<TGridIn, TGridOut, LabMPI>(parser);
    else if (operator_name == "Smoother")
        gridmanip = new Smoother<TGridIn, TGridOut, LabMPI>(parser);
    else {
        if (isroot)
            std::cerr << "ERROR: Undefined operator '" << operator_name << "'"
                      << std::endl;
        abort();
    }
    if (isroot)
        parser.print_args();
    (*gridmanip)(*grid_in, *grid_out, isroot);

    // save manipulations
    const std::string outfile = parser("out_file").asString("scalar_out");
    outfactory(*grid_out, outfile);

    MPI_Barrier(MPI_COMM_WORLD);

    // clean up
    delete grid_in;
    delete grid_out;
    delete gridmanip;

    MPI_Finalize();
    return 0;
}
