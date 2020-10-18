//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Written by Guido Novati (novatig@ethz.ch).
//
#include "Simulation.h"
#include "obstacles/ObstacleVector.h"

#include "operators/AdvectionDiffusion.h"
#include "operators/Checkpoint.h"
#include "operators/ComputeDissipation.h"
#include "operators/ExternalForcing.h"
#include "operators/FadeOut.h"
#include "operators/FluidSolidForces.h"
#include "operators/InitialConditions.h"
#include "operators/ObstaclesCreate.h"
#include "operators/ObstaclesUpdate.h"
#include "operators/Penalization.h"
#include "operators/PressureProjection.h"
#include "operators/IterativePressureNonUniform.h"
#include "operators/IterativePressurePenalization.h"
#include "operators/PressureRHS.h"
#include "operators/FixedMassFlux_nonUniform.h"
#include "operators/SGS.h"
#include "operators/Analysis.h"
#include "operators/HITfiltering.h"

#include "spectralOperators/SpectralForcing.h"

#include "obstacles/ObstacleFactory.h"
#include "operators/ProcessHelpers.h"
#include "utils/NonUniformScheme.h"

#include <Cubism/HDF5Dumper_MPI.h>
#include <Cubism/HDF5SliceDumperMPI.h>
#include <Cubism/MeshKernels.h>

#include <iomanip>
#include <iostream>
#include <sstream>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

#ifdef CUP_ASYNC_DUMP
 using SliceType  = cubism::SliceTypesMPI::Slice<DumpGridMPI>;
#else
 using SliceType  = cubism::SliceTypesMPI::Slice<FluidGridMPI>;
#endif

// Initialization from cmdline arguments is done in few steps, because grid has
// to be created before the obstacles and slices are created.
Simulation::Simulation(const SimulationData &_sim) : sim(_sim)
{
  sim._preprocessArguments();

  // Grid has to be initialized before slices and obstacles.
  setupGrid(nullptr);

  // Define an empty obstacle vector, which can be later modified.
  sim.obstacle_vector = new ObstacleVector(sim);

  _init(false);
}

Simulation::Simulation(MPI_Comm mpicomm) : sim(mpicomm)
{
  // What about setupGrid and other stuff?
  sim.obstacle_vector = new ObstacleVector(sim);
  _init(false);
}

Simulation::Simulation(MPI_Comm mpicomm, ArgumentParser & parser)
    : sim(mpicomm, parser)
{
  sim._preprocessArguments();

  // Grid has to be initialized before slices and obstacles.
  setupGrid(&parser);

  #ifdef CUP_ASYNC_DUMP
    sim.m_slices = SliceType::getEntities<SliceType>(parser, * sim.dump);
  #else
    sim.m_slices = SliceType::getEntities<SliceType>(parser, * sim.grid);
  #endif

  // ========== OBSTACLES ==========
  sim.obstacle_vector = new ObstacleVector(sim);
  ObstacleFactory(sim).addObstacles(parser);

  const bool bRestart = parser("-restart").asBool(false);
  _init(bRestart);
}

const std::vector<std::shared_ptr<Obstacle>>& Simulation::getObstacleVector() const
{
    return sim.obstacle_vector->getObstacleVector();
}

/* DEPRECATED. Keep until `source/bindings/Simulation.cpp` is fixed.

// For Python bindings. Really no need for `const` here...
Simulation::Simulation(
  std::array<int,3> cells, std::array<int, 3> nproc, MPI_Comm comm, int nsteps,
  double endTime, double nu, double CFL, double lambda, double DLM,
  std::array<double,3> uinf, bool verbose, int freqDiagnostics,
  bool b3Ddump, bool b2Ddump, double fadeOutLength, int saveFreq,
  double saveTime, const std::string &path4serialization, bool restart) :
  sim(comm)
{
  sim.nprocsx = nproc[0];
  sim.nprocsy = nproc[1];
  sim.nprocsz = nproc[2];
  sim.nsteps = nsteps;
  sim.endTime = endTime;
  sim.uinf[0] = uinf[0];
  sim.uinf[1] = uinf[1];
  sim.uinf[2] = uinf[2];
  sim.nu = nu;
  sim.CFL = CFL;
  sim.lambda = lambda;
  sim.DLM = DLM;
  sim.verbose = verbose;
  sim.freqDiagnostics = freqDiagnostics;
  sim.b3Ddump = b3Ddump;
  sim.b2Ddump = b2Ddump;
  sim.fadeOutLengthPRHS[0] = fadeOutLength;
  sim.fadeOutLengthPRHS[1] = fadeOutLength;
  sim.fadeOutLengthPRHS[2] = fadeOutLength;
  sim.fadeOutLengthU[0] = fadeOutLength;
  sim.fadeOutLengthU[1] = fadeOutLength;
  sim.fadeOutLengthU[2] = fadeOutLength;
  sim.saveFreq = saveFreq;
  sim.saveTime = saveTime;
  sim.path4serialization = path4serialization;

  if (cells[0] < 0 || cells[1] < 0 || cells[2] < 0)
    throw std::invalid_argument("N. of cells not provided.");
  if (   cells[0] % FluidBlock::sizeX != 0
      || cells[1] % FluidBlock::sizeY != 0
      || cells[2] % FluidBlock::sizeZ != 0 )
    throw std::invalid_argument("N. of cells must be multiple of block size.");

  sim.bpdx = cells[0] / FluidBlock::sizeX;
  sim.bpdy = cells[1] / FluidBlock::sizeY;
  sim.bpdz = cells[2] / FluidBlock::sizeZ;
  sim._preprocessArguments();
  setupGrid();  // Grid has to be initialized before slices and obstacles.
  setObstacleVector(new ObstacleVector(sim));
  _init(restart);
}
*/

void Simulation::_init(const bool restart)
{
  setupOperators();

  if (restart)
    _deserialize();
  else if (sim.icFromH5 != "")
    _icFromH5(sim.icFromH5);
  else
    _ic();
  MPI_Barrier(sim.app_comm);
  //_serialize("init");

  assert(sim.obstacle_vector != nullptr);
  if (sim.rank == 0)
  {
    const double maxU = std::max({sim.uinf[0], sim.uinf[1], sim.uinf[2]});
    const double length = sim.obstacle_vector->getD();
    const double re = length * std::max(maxU, length) / sim.nu;
    assert(length > 0 || sim.obstacle_vector->getObstacleVector().empty());
    printf("Kinematic viscosity:%f, Re:%f, length scale:%f\n",sim.nu,re,length);
  }
}


void Simulation::reset()
{
  if (sim.icFromH5 != "") _icFromH5(sim.icFromH5);
  else _ic();

  sim.nextSaveTime = 0;
  sim.step = 0; sim.time = 0;
  sim.uinf = std::array<Real,3> {{ (Real)0, (Real)0, (Real)0 }};
  //sim.obstacle_vector->reset(); // TODO
  if(sim.obstacle_vector->nObstacles() > 0) {
    printf("TODO Implement reset also for obstacles if needed!\n");
    fflush(0); MPI_Abort(sim.app_comm, 1);
  }
}

void Simulation::_ic()
{
  InitialConditions coordIC(sim);
  sim.startProfiler(coordIC.getName());
  coordIC(0);
  sim.stopProfiler();
}

void Simulation::_icFromH5(std::string h5File)
{
  if (sim.rank==0) std::cout << "Extracting Initial Conditions from " << h5File << std::endl;

  #ifdef CUBISM_USE_HDF
    ReadHDF5_MPI<StreamerVelocityVector, DumpReal>(* sim.grid,
      h5File, sim.path4serialization);
  #else
    printf("Unable to restart without  HDF5 library. Aborting...\n");
    fflush(0); MPI_Abort(sim.grid->getCartComm(), 1);
  #endif

  sim.obstacle_vector->restart(sim.path4serialization+"/"+sim.icFromH5);

  // prepare time for next save
  sim.nextSaveTime = sim.time + sim.saveTime;
  MPI_Barrier(sim.app_comm);
}

void Simulation::setupGrid(cubism::ArgumentParser *parser_ptr)
{
  if( not sim.bUseStretchedGrid )
  {
    if(sim.rank==0)
      printf("Uniform-resolution grid of sizes: %f %f %f\n",
      sim.extent[0],sim.extent[1],sim.extent[2]);
    sim.grid = new FluidGridMPI(sim.nprocsx,sim.nprocsy,sim.nprocsz,
                                sim.local_bpdx,sim.local_bpdy,sim.local_bpdz,
                                sim.maxextent, sim.app_comm);
    assert(sim.grid != nullptr);

    #ifdef CUP_ASYNC_DUMP
      // create new comm so that if there is a barrier main work is not affected
      MPI_Comm_split(sim.app_comm, 0, sim.rank, &sim.dump_comm);
      sim.dump = new  DumpGridMPI(sim.nprocsx,sim.nprocsy,sim.nprocsz,
                                  sim.local_bpdx,sim.local_bpdy,sim.local_bpdz,
                                  sim.maxextent, sim.dump_comm);
    #endif
    sim.hmin  = sim.grid->getBlocksInfo()[0].h_gridpoint;
    sim.hmax  = sim.grid->getBlocksInfo()[0].h_gridpoint;
    sim.hmean = sim.grid->getBlocksInfo()[0].h_gridpoint;
  }
  else
  {
    if(sim.rank==0)
      printf("Stretched grid of sizes: %f %f %f\n",
      sim.extent[0],sim.extent[1],sim.extent[2]);
    NonUniformScheme<FluidBlock> * nonuniform = nullptr;
    nonuniform = new NonUniformScheme<FluidBlock>( 0, sim.extent[0],
      0, sim.extent[1], 0, sim.extent[2], sim.bpdx, sim.bpdy, sim.bpdz);
    assert(nonuniform not_eq nullptr);
    // initialize scheme
    assert(parser_ptr != nullptr
           && "Cannot use a stretched grid and initialize without an ArgumentParser.");
    MeshDensityFactory mk(* parser_ptr);
    nonuniform->init(mk.get_mesh_kernel(0), mk.get_mesh_kernel(1), mk.get_mesh_kernel(2));

    sim.grid = new FluidGridMPI(
      & nonuniform->get_map_x(), & nonuniform->get_map_y(),
      & nonuniform->get_map_z(), sim.nprocsx, sim.nprocsy, sim.nprocsz,
      sim.local_bpdx, sim.local_bpdy, sim.local_bpdz, sim.app_comm);
    assert(sim.grid != nullptr);

    #ifdef CUP_ASYNC_DUMP
      // create new comm so that if there is a barrier main work is not affected
      MPI_Comm_split(sim.app_comm, 0, sim.rank, & sim.dump_comm);
      NonUniformScheme<DumpBlock> * dump_nonuniform = nullptr;
      dump_nonuniform = new NonUniformScheme<DumpBlock>( 0, sim.extent[0],
        0, sim.extent[1], 0, sim.extent[2], sim.bpdx, sim.bpdy, sim.bpdz);
      dump_nonuniform->init(mk.get_mesh_kernel(0), mk.get_mesh_kernel(1), mk.get_mesh_kernel(2));
      sim.dump = new  DumpGridMPI(
        & dump_nonuniform->get_map_x(), & dump_nonuniform->get_map_y(),
        & dump_nonuniform->get_map_z(), sim.nprocsx,sim.nprocsy,sim.nprocsz,
        sim.local_bpdx, sim.local_bpdy, sim.local_bpdz, sim.dump_comm);
      sim.dump_nonuniform = (void *) dump_nonuniform; // to delete it at the end
    #endif

    // setp block coefficients
    nonuniform->template setup_coefficients<FDcoeffs_2ndOrder>(
      sim.grid->getBlocksInfo());
    nonuniform->template setup_coefficients<FDcoeffs_4thOrder>(
      sim.grid->getBlocksInfo(), true);
    //nonuniform->setup_inverse_spacing(
    //  sim.grid->getBlocksInfo());

    // some statistics
    nonuniform->print_mesh_statistics(sim.rank == 0);
    sim.hmin  = nonuniform->minimum_cell_width();
    sim.hmax  = nonuniform->maximum_cell_width();
    sim.hmean = nonuniform->compute_mean_grid_spacing();
    printf("hmin:%e hmax:%e hmean:%e\n", sim.hmin, sim.hmax, sim.hmean);
    sim.nonuniform = (void *) nonuniform; // to delete it at the end
  }

  const std::vector<BlockInfo>& vInfo = sim.vInfo();
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<vInfo.size(); i++) {
    FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
    b.min_pos = vInfo[i].pos<Real>(0, 0, 0);
    b.max_pos = vInfo[i].pos<Real>(FluidBlock::sizeX-1,
                                   FluidBlock::sizeY-1,
                                   FluidBlock::sizeZ-1);
  }
}

void Simulation::setupOperators()
{
  sim.pipeline.clear();
  // Do not change order of operations without explicit permission from Guido

  // Obstacle shape has to be known already here.
  sim.pipeline.push_back(
      checkpointPreObstacles = new Checkpoint(sim, "PreObstacles"));

  // Creates the char function, sdf, and def vel for all obstacles at the curr
  // timestep. At this point we do NOT know the translation and rot vel of the
  // obstacles. We need to solve implicit system when the pre-penalization vel
  // is finalized on the grid.
  // Here we also compute obstacles' centres of mass which are computed from
  // the char func on the grid. This is different from "position" which is
  // the quantity that is advected and used to construct shape.
  Operator *createObstacles = new CreateObstacles(sim);
  sim.pipeline.push_back(createObstacles);

  // Performs:
  // \tilde{u} = u_t + \delta t (\nu \nabla^2 u_t - (u_t \cdot \nabla) u_t )
  sim.pipeline.push_back(new AdvectionDiffusion(sim));

  if (sim.sgs != "")
    sim.pipeline.push_back(new SGS(sim));

  // Used to add an uniform pressure gradient / uniform driving force.
  // If the force were space-varying then we would need to include in the
  // pressure equation's RHS.
  if(   sim.uMax_forced > 0
     && sim.initCond not_eq "taylorGreen"  // also uses sim.uMax_forced param
     && sim.bChannelFixedMassFlux == false) // also uses sim.uMax_forced param
    sim.pipeline.push_back(new ExternalForcing(sim));

  if(sim.bIterativePenalization)
  {
    if(sim.bUseStretchedGrid && sim.obstacle_vector->nObstacles() > 0) {
      printf("Non-uniform grids AND obstacles AND iterative penalization are not compatible. Pick 2 out of 3.\n");
      fflush(0); MPI_Abort(sim.app_comm, 1);
    }
    if(sim.bUseStretchedGrid)
      sim.pipeline.push_back(new IterativePressureNonUniform(sim));
    else if (sim.obstacle_vector->nObstacles() > 0)
      sim.pipeline.push_back(new IterativePressurePenalization(sim));
    else {
      printf("Undefined type of pressure iteration. What are ya simulating?\n");
      fflush(0); MPI_Abort(sim.app_comm, 1);
    }
  }
  else
  {
    #ifdef PENAL_THEN_PRES
      // Compute velocity of the obstacles and, in the same sweep if frame of ref
      // is moving, we update uinf. Requires the pre-penal vel field on the grid!!
      // We also update position and quaternions of the obstacles.
      sim.pipeline.push_back(new UpdateObstacles(sim));

      // With pre-penal vel field and obstacles' velocities perform penalization.
      sim.pipeline.push_back(new Penalization(sim));
    #endif
    // Places Udef on the grid and computes the RHS of the Poisson Eq
    // overwrites tmpU, tmpV, tmpW and pressure solver's RHS
    // places in press RHS = (1 - X) \nabla \cdot u_f
    sim.pipeline.push_back(new PressureRHS(sim));

    // Solves the Poisson Eq to get the pressure and finalizes the velocity
    // u_{t+1} = \tilde{u} -\delta t \nabla P. This is final pre-penal vel field.
    sim.pipeline.push_back(new PressureProjection(sim));

    #ifndef PENAL_THEN_PRES
      sim.pipeline.push_back(new UpdateObstacles(sim));
      sim.pipeline.push_back(new Penalization(sim));
    #endif
  }

  if (sim.spectralForcing)
    sim.pipeline.push_back(new SpectralForcing(sim));

  // With finalized velocity and pressure, compute forces and dissipation
  sim.pipeline.push_back(new ComputeForces(sim));

  sim.pipeline.push_back(new ComputeDissipation(sim));

  if (sim.bChannelFixedMassFlux)
    sim.pipeline.push_back(new FixedMassFlux_nonUniform(sim));

  // At this point the velocity computation is finished.
  sim.pipeline.push_back(
      checkpointPostVelocity = new Checkpoint(sim, "PostVelocity"));

  //sim.pipeline.push_back(new HITfiltering(sim));
  sim.pipeline.push_back(new StructureFunctions(sim));

  sim.pipeline.push_back(new Analysis(sim));

  if(sim.rank==0) {
    printf("Coordinator/Operator ordering:\n");
    for (size_t c=0; c<sim.pipeline.size(); c++)
      printf("\t%s\n", sim.pipeline[c]->getName().c_str());
  }
  //immediately call create!
  (*createObstacles)(0);
}

double Simulation::calcMaxTimestep()
{
  assert(sim.grid not_eq nullptr);
  const double hMin = sim.hmin, CFL = sim.CFL;
  sim.uMax_measured = sim.bKeepMomentumConstant? findMaxUzeroMom(sim)
                                               : findMaxU(sim);
  const double dtDif = hMin * hMin / sim.nu;
  const double dtAdv = hMin / ( sim.uMax_measured + 1e-8 );
  sim.dt = CFL * std::min(dtDif, dtAdv);
  if ( sim.step < sim.rampup )
  {
    const double x = sim.step / (double) sim.rampup;
    const double rampCFL = std::exp(std::log(1e-3)*(1-x) + std::log(CFL)*x);
    sim.dt = rampCFL * std::min(dtDif, dtAdv);
  }
  // if DLM>0, adapt lambda such that penal term is independent of time step
  if (sim.DLM > 0) sim.lambda = sim.DLM / sim.dt;
  if (sim.verbose)
    printf("maxU:%f minH:%f dtF:%e dtC:%e dt:%e lambda:%e\n",
      sim.uMax_measured, hMin, dtDif, dtAdv, sim.dt, sim.lambda);
  return sim.dt;
}

void Simulation::_serialize(const std::string append)
{
  std::stringstream ssR;
  if (append == "") ssR<<"restart_";
  else ssR<<append;
  ssR<<std::setfill('0')<<std::setw(9)<<sim.step;
  const std::string fpath = sim.path4serialization + "/" + ssR.str();
  if(sim.rank==0) std::cout<<"Saving to "<<fpath<<"\n";

  if (sim.rank==0) { //rank 0 saves step id and obstacles
    sim.obstacle_vector->save(fpath);
    //safety status in case of crash/timeout during grid save:
    FILE * f = fopen((fpath+".status").c_str(), "w");
    assert(f != NULL);
    fprintf(f, "time: %20.20e\n", sim.time);
    fprintf(f, "stepid: %d\n", (int)sim.step);
    fprintf(f, "uinfx: %20.20e\n", sim.uinf[0]);
    fprintf(f, "uinfy: %20.20e\n", sim.uinf[1]);
    fprintf(f, "uinfz: %20.20e\n", sim.uinf[2]);
    fclose(f);
  }

  // hack to write sdf ... on pressure field. only debug.
  //const auto vecOB = sim.obstacle_vector->getAllObstacleBlocks();
  //putSDFonGrid(sim.vInfo(), vecOB);

  #ifdef CUBISM_USE_HDF
  std::stringstream ssF;
  if (append == "")
   ssF<<"avemaria_"<<std::setfill('0')<<std::setw(9)<<sim.step;
  else
   ssF<<"2D_"<<append<<std::setfill('0')<<std::setw(9)<<sim.step;

  #ifdef CUP_ASYNC_DUMP
    // if a thread was already created, make sure it has finished
    if(sim.dumper not_eq nullptr) {
      sim.dumper->join();
      delete sim.dumper;
      sim.dumper = nullptr;
    }
    // copy qois from grid to dump
    copyDumpGrid(* sim.grid, * sim.dump);
    const auto * const grid2Dump = sim.dump;
  #else //CUP_ASYNC_DUMP
    const auto * const grid2Dump = sim.grid;
  #endif //CUP_ASYNC_DUMP

  const auto name3d = ssR.str(), name2d = ssF.str(); // sstreams are weird

  const auto dumpFunction = [=] () {
    if(sim.b2Ddump) {
      int sliceIdx = 0;
      for (const auto& slice : sim.m_slices) {
        const std::string slicespec = "slice_"+std::to_string(sliceIdx++)+"_";
        const auto nameV = slicespec + StreamerVelocityVector::prefix() +name2d;
        const auto nameP = slicespec + StreamerPressure::prefix() +name2d;
        const auto nameX = slicespec + StreamerChi::prefix() +name2d;
        DumpSliceHDF5MPI<StreamerVelocityVector, DumpReal>(
          slice, sim.time, nameV, sim.path4serialization);
        DumpSliceHDF5MPI<StreamerPressure, DumpReal>(
          slice, sim.time, nameP, sim.path4serialization);
        DumpSliceHDF5MPI<StreamerChi, DumpReal>(
          slice, sim.time, nameX, sim.path4serialization);
      }
    }
    if(sim.b3Ddump) {
      const std::string nameV = StreamerVelocityVector::prefix()+name3d;
      const std::string nameP = StreamerPressure::prefix()+name3d;
      const std::string nameX = StreamerChi::prefix()+name3d;
      DumpHDF5_MPI<StreamerVelocityVector, DumpReal>(
        *grid2Dump, sim.time, nameV, sim.path4serialization);
      DumpHDF5_MPI<StreamerPressure, DumpReal>(
        *grid2Dump, sim.time, nameP, sim.path4serialization);
      DumpHDF5_MPI<StreamerChi, DumpReal>(
        *grid2Dump, sim.time, nameX, sim.path4serialization);
    }
  };

  #ifdef CUP_ASYNC_DUMP
    sim.dumper = new std::thread( dumpFunction );
  #else //CUP_ASYNC_DUMP
    dumpFunction();
  #endif //CUP_ASYNC_DUMP
  #endif //CUBISM_USE_HDF


  if(sim.rank==0) { //saved the grid! Write status to remember most recent save
    std::string restart_status = sim.path4serialization+"/restart.status";
    FILE * f = fopen(restart_status.c_str(), "w");
    assert(f != NULL);
    fprintf(f, "time: %20.20e\n", sim.time);
    fprintf(f, "stepid: %d\n", (int)sim.step);
    fprintf(f, "uinfx: %20.20e\n", sim.uinf[0]);
    fprintf(f, "uinfy: %20.20e\n", sim.uinf[1]);
    fprintf(f, "uinfz: %20.20e\n", sim.uinf[2]);
    fclose(f);
    printf("time:  %20.20e\n", sim.time);
    printf("stepid: %d\n", (int)sim.step);
    printf("uinfx: %20.20e\n", sim.uinf[0]);
    printf("uinfy: %20.20e\n", sim.uinf[1]);
    printf("uinfz: %20.20e\n", sim.uinf[2]);
  }

  //CoordinatorDiagnostics coordDiags(grid,time,step);
  //coordDiags(dt);
  //obstacle_vector->interpolateOnSkin(time, step);
}

void Simulation::_deserialize()
{
  {
    std::string restartfile = sim.path4serialization+"/restart.status";
    FILE * f = fopen(restartfile.c_str(), "r");
    if (f == NULL) {
      printf("Could not restart... starting a new sim.\n");
      return;
    }
    assert(f != NULL);
    bool ret = true;
    ret = ret && 1==fscanf(f, "time: %le\n",   &sim.time);
    ret = ret && 1==fscanf(f, "stepid: %d\n", &sim.step);
    #ifndef CUP_SINGLE_PRECISION
    ret = ret && 1==fscanf(f, "uinfx: %le\n", &sim.uinf[0]);
    ret = ret && 1==fscanf(f, "uinfy: %le\n", &sim.uinf[1]);
    ret = ret && 1==fscanf(f, "uinfz: %le\n", &sim.uinf[2]);
    #else // CUP_SINGLE_PRECISION
    ret = ret && 1==fscanf(f, "uinfx: %e\n", &sim.uinf[0]);
    ret = ret && 1==fscanf(f, "uinfy: %e\n", &sim.uinf[1]);
    ret = ret && 1==fscanf(f, "uinfz: %e\n", &sim.uinf[2]);
    #endif // CUP_SINGLE_PRECISION
    fclose(f);
    if( (not ret) || sim.step<0 || sim.time<0) {
      printf("Error reading restart file. Aborting...\n");
      fflush(0); MPI_Abort(sim.grid->getCartComm(), 1);
    }
  }

  std::stringstream ssR;
  ssR<<"restart_"<<std::setfill('0')<<std::setw(9)<<sim.step;
  if (sim.rank==0) std::cout << "Restarting from " << ssR.str() << "\n";

  #ifdef CUBISM_USE_HDF
    ReadHDF5_MPI<StreamerVelocityVector, DumpReal>(* sim.grid,
      StreamerVelocityVector::prefix()+ssR.str(), sim.path4serialization);
  #else
    printf("Unable to restart without  HDF5 library. Aborting...\n");
    fflush(0); MPI_Abort(sim.grid->getCartComm(), 1);
  #endif

  sim.obstacle_vector->restart(sim.path4serialization+"/"+ssR.str());

  printf("DESERIALIZATION: time is %f and step id is %d\n", sim.time, sim.step);
  // prepare time for next save
  sim.nextSaveTime = sim.time + sim.saveTime;
}

void Simulation::run()
{
  for (;;) {
    sim.startProfiler("DT");
    const double dt = calcMaxTimestep();
    sim.stopProfiler();

    if (timestep(dt)) break;
  }
}

bool Simulation::timestep(const double dt)
{
    const bool bDumpFreq = (sim.saveFreq>0 && (sim.step+ 1)%sim.saveFreq==0);
    const bool bDumpTime = (sim.saveTime>0 && (sim.time+dt)>sim.nextSaveTime);
    if (bDumpTime) sim.nextSaveTime += sim.saveTime;
    sim.bDump = (bDumpFreq || bDumpTime);

    for (size_t c=0; c<sim.pipeline.size(); c++) {
      (*sim.pipeline[c])(dt);
      //_serialize(sim.pipeline[c]->getName()+std::to_string(sim.step));
    }
    sim.step++;
    sim.time+=dt;

    if(sim.verbose) printf("%d : %e uInf {%f %f %f}\n",
      sim.step,sim.time,sim.uinf[0],sim.uinf[1],sim.uinf[2]);

    sim.startProfiler("Save");
    if( sim.bDump ) _serialize();
    sim.stopProfiler();

    if (sim.step % 50 == 0 && sim.verbose) sim.printResetProfiler();
    if ((sim.endTime>0 && sim.time>sim.endTime) ||
        (sim.nsteps!=0 && sim.step>=sim.nsteps) ) {
      if(sim.verbose)
        std::cout<<"Finished at time "<<sim.time<<" in "<<sim.step<<" steps.\n";
      return true;  // Finished.
    }

    return false;  // Not yet finished.
}

CubismUP_3D_NAMESPACE_END
