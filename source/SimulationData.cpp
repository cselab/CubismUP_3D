//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include <unistd.h>

#include "SimulationData.h"
#include "operators/Operator.h"
#include "obstacles/ObstacleVector.h"
#include "utils/NonUniformScheme.h"

#include <Cubism/ArgumentParser.h>
#include <Cubism/Profiler.h>
#include <Cubism/HDF5SliceDumperMPI.h>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

SimulationData::SimulationData(const SimulationData &) = default;
SimulationData::SimulationData(SimulationData &&) = default;
SimulationData::SimulationData(MPI_Comm mpicomm) :
  app_comm(mpicomm)
{
  MPI_Comm_rank(app_comm, &rank);
  MPI_Comm_size(app_comm, &nprocs);
}

SimulationData::SimulationData(MPI_Comm mpicomm, ArgumentParser &parser)
    : SimulationData(mpicomm)
{
  if (rank == 0) parser.print_args();

  // ========== SIMULATION ==========
  // GRID
  bpdx = parser("-bpdx").asInt();
  bpdy = parser("-bpdy").asInt();
  bpdz = parser("-bpdz").asInt();
  nprocsx = parser("-nprocsx").asInt(-1);
  nprocsy = parser("-nprocsy").asInt(-1);
  nprocsz = parser("-nprocsz").asInt(-1);
  extent[0] = parser("extentx").asDouble(1);
  extent[1] = parser("extenty").asDouble(0);
  extent[2] = parser("extentz").asDouble(0);

  // FLOW
  nu = parser("-nu").asDouble();

  // IC
  initCond = parser("-initCond").asString("zero");
  spectralIC = parser("-spectralIC").asString("");
  k0 = parser("-k0").asDouble(10.0);
  tke0 = parser("-tke0").asDouble(1.0);
  icFromH5 = parser("-icFromH5").asString("");

  // FORCING HIT=
  turbKinEn_target = parser("-turbKinEn_target").asDouble(0);
  enInjectionRate = parser("-energyInjectionRate").asDouble(0);
  const bool bSpectralForcingHint = turbKinEn_target>0 || enInjectionRate>0;
  spectralForcing = parser("-spectralForcing").asBool(bSpectralForcingHint);
  if(turbKinEn_target>0 && enInjectionRate>0) {
    fprintf(stderr,"ERROR: either constant energy injection rate "
                   "or forcing to fixed energy target\n");
    fflush(0); abort();
  }

  // PIPELINE && FORCING
  freqDiagnostics = parser("-freqDiagnostics").asInt(100);
  bIterativePenalization = parser("-iterativePenalization").asBool(false);
  bImplicitPenalization = parser("-implicitPenalization").asBool(false);
  bKeepMomentumConstant = parser("-keepMomentumConstant").asBool(false);
  bChannelFixedMassFlux = parser("-channelFixedMassFlux").asBool(false);

  bRungeKutta23 = parser("-RungeKutta23").asBool(false);
  bAdvection3rdOrder = parser("-Advection3rdOrder").asBool(true);

  uMax_forced = parser("-uMax_forced").asDouble(0.0);

  // SGS
  sgs = parser("-sgs").asString("");
  cs = parser("-cs").asDouble(0.2);
  bComputeCs2Spectrum = parser("-cs2spectrum").asBool(false);

  // SGS_RL
  sgs_rl = parser("-sgs_rl").asBool(false);
  nAgentsPerBlock = parser("-nAgentsPerBlock").asInt(1);

  lambda = parser("-lambda").asDouble(1e6);
  DLM = parser("-use-dlm").asDouble(0);
  CFL = parser("-CFL").asDouble(.1);
  uinf[0] = parser("-uinfx").asDouble(0.0);
  uinf[1] = parser("-uinfy").asDouble(0.0);
  uinf[2] = parser("-uinfz").asDouble(0.0);

  // OUTPUT
  verbose = parser("-verbose").asBool(true) && rank == 0;
  b2Ddump = parser("-dump2D").asBool(false);
  b3Ddump = parser("-dump3D").asBool(true);

  // ANALYSIS
  analysis = parser("-analysis").asString("");
  timeAnalysis = parser("-tAnalysis").asDouble(0.0);
  freqAnalysis = parser("-fAnalysis").asInt(0);

  int dumpFreq = parser("-fdump").asDouble(0);       // dumpFreq==0 means dump freq (in #steps) is not active
  double dumpTime = parser("-tdump").asDouble(0.0);  // dumpTime==0 means dump freq (in time)   is not active
  saveFreq = parser("-fsave").asInt(0);         // dumpFreq==0 means dump freq (in #steps) is not active
  saveTime = parser("-tsave").asDouble(0.0);    // dumpTime==0 means dump freq (in time)   is not active
  rampup = parser("-rampup").asInt(100); // number of dt ramp-up steps

  nsteps = parser("-nsteps").asInt(0);    // 0 to disable this stopping critera.
  endTime = parser("-tend").asDouble(0);  // 0 to disable this stopping critera.

  // TEMP: Removed distinction saving-dumping. Backward compatibility:
  if (saveFreq <= 0 && dumpFreq > 0) saveFreq = dumpFreq;
  if (saveTime <= 0 && dumpTime > 0) saveTime = dumpTime;

  path4serialization = parser("-serialization").asString("./");

  // INITIALIZATION: Mostly unused
  useSolver = parser("-useSolver").asString("");
  // BOUNDARY CONDITIONS
  // accepted dirichlet, periodic, freespace/unbounded, fakeOpen
  std::string BC_x = parser("-BC_x").asString("dirichlet");
  std::string BC_y = parser("-BC_y").asString("dirichlet");
  std::string BC_z = parser("-BC_z").asString("dirichlet");
  const Real fadeLen = parser("-fade_len").asDouble(0.0);
  // BC
  if(BC_x=="unbounded") BC_x = "freespace"; // tomato tomato
  if(BC_y=="unbounded") BC_y = "freespace"; // tomato tomato
  if(BC_z=="unbounded") BC_z = "freespace"; // tomato tomato
  // boundary killing useless for unbounded or periodic
  fadeOutLengthPRHS[0] = BC_x=="dirichlet"? fadeLen : 0;
  fadeOutLengthPRHS[1] = BC_y=="dirichlet"? fadeLen : 0;
  fadeOutLengthPRHS[2] = BC_z=="dirichlet"? fadeLen : 0;

  if(BC_x=="freespace" || BC_y=="freespace" || BC_z=="freespace")
  {
    if(BC_x=="freespace" && BC_y=="freespace" && BC_z=="freespace") {
      bUseUnboundedBC = true; // poisson solver
    } else {
     fprintf(stderr,"ERROR: either all or no BC can be freespace/unbounded!\n");
     fflush(0); abort();
    }
  }

  if(BC_x=="fakeopen" || BC_y=="fakeopen" || BC_z=="fakeopen")
  {
    if(BC_x=="fakeopen" && BC_y=="fakeopen" && BC_z=="fakeopen")
    {
      fadeOutLengthU[0] = fadeLen; fadeOutLengthPRHS[0] = fadeLen;
      fadeOutLengthU[1] = fadeLen; fadeOutLengthPRHS[1] = fadeLen;
      fadeOutLengthU[2] = fadeLen; fadeOutLengthPRHS[2] = fadeLen;
      BC_x = "freespace"; BC_y = "freespace"; BC_z = "freespace";
      bUseFourierBC = true; // poisson solver
    } else {
     fprintf(stderr,"ERROR: either all or no BC can be fakeopen!\n");
     fflush(0); abort();
    }
  }

  BCx_flag = string2BCflag(BC_x);
  BCy_flag = string2BCflag(BC_y);
  BCz_flag = string2BCflag(BC_z);

  // DFT if we are periodic in all directions:
  if(BC_x=="periodic"&&BC_y=="periodic"&&BC_z=="periodic") bUseFourierBC = true;
  if(rank==0)
  printf("Boundary pressure RHS / FD smoothing region sizes {%f,%f,%f}\n",
    fadeOutLengthPRHS[0], fadeOutLengthPRHS[1], fadeOutLengthPRHS[2]);

  // ============ REST =============
}

void SimulationData::_preprocessArguments()
{
  assert(profiler == nullptr);  // This should not be possible at all.
  profiler = new cubism::Profiler();

  // Grid.
  if (bpdx < 1 || bpdy < 1 || bpdz < 1) {
      fprintf(stderr, "Invalid bpd: %d x %d x %d\n", bpdx, bpdy, bpdz);
      fflush(0); abort();
  }
  const double NFE[3] = {
      (double) bpdx * FluidBlock::sizeX,
      (double) bpdy * FluidBlock::sizeY,
      (double) bpdz * FluidBlock::sizeZ,
  };
  const double maxbpd = std::max({NFE[0], NFE[1], NFE[2]});
  maxextent = std::max({extent[0], extent[1], extent[2]});
  if( extent[0] <= 0 || extent[1] <= 0 || extent[2] <= 0 ) {
    bUseStretchedGrid = false;
    extent[0] = (NFE[0]/maxbpd) * maxextent;
    extent[1] = (NFE[1]/maxbpd) * maxextent;
    extent[2] = (NFE[2]/maxbpd) * maxextent;
  } else {
    bUseStretchedGrid = true;
  }
  printf("Domain extent: %lg %lg %lg\n", extent[0], extent[1], extent[2]);

  // Flow.
  assert(nu >= 0);
  assert(lambda > 0 || DLM > 0);
  assert(CFL > 0.0);

  // Output.
  assert(saveFreq >= 0.0);
  assert(saveTime >= 0.0);

  // MPI.
  if (nprocsy <= 0) nprocsy = 1;
  if (nprocsz <= 0) nprocsz = 1;
  if (nprocsx <= 0) nprocsx = nprocs / nprocsy / nprocsz;

  if (nprocsx * nprocsy * nprocsz != nprocs) {
    fprintf(stderr, "Invalid domain decomposition. %d x %d x %d != %d!\n",
            nprocsx, nprocsy, nprocsz, nprocs);
    fflush(0); MPI_Abort(app_comm, 1);
  }

  if ( bpdx % nprocsx != 0 ||
       bpdy % nprocsy != 0 ||
       bpdz % nprocsz != 0   ) {
    printf("Incompatible domain decomposition: bpd*/nproc* should be an integer");
    fflush(0); MPI_Abort(app_comm, 1);
  }

  local_bpdx = bpdx / nprocsx;
  local_bpdy = bpdy / nprocsy;
  local_bpdz = bpdz / nprocsz;

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  const int nthreads = omp_get_max_threads();
  printf("Rank %d (of %d) with %d threads on host Hostname: %s\n",
          rank, nprocs, nthreads, hostname);
  //if (communicator not_eq nullptr) //Yo dawg I heard you like communicators.
  //  communicator->comm_MPI = grid->getCartComm();
  if(rank==0) {
    printf("Blocks per dimension: [%d %d %d]\n",bpdx,bpdy,bpdz);
    printf("Nranks per dimension: [%d %d %d]\n",nprocsx,nprocsy,nprocsz);
    printf("Local blocks per dimension: [%d %d %d]\n",
      local_bpdx,local_bpdy,local_bpdz);
  }
  fflush(0);
}

SimulationData::~SimulationData()
{
  delete grid;
  delete profiler;
  delete obstacle_vector;
  if(nonuniform not_eq nullptr) {
    NonUniformScheme<FluidBlock>* nonuniform_ = static_cast<NonUniformScheme<FluidBlock>*>(nonuniform);
    assert(nonuniform_ not_eq nullptr);
    delete nonuniform_;
  }
  while(!pipeline.empty()) {
    auto * g = pipeline.back();
    pipeline.pop_back();
    delete g;
  }
  #ifdef CUP_ASYNC_DUMP
    if(dumper not_eq nullptr) {
      dumper->join();
      delete dumper;
    }
    delete dump;
    if(dump_nonuniform not_eq nullptr) {
      NonUniformScheme<DumpBlock>* dump_nonuniform_ = static_cast<NonUniformScheme<DumpBlock>*>(dump_nonuniform);
      assert(dump_nonuniform_ not_eq nullptr);
      delete dump_nonuniform_;
    }
    if (dump_comm != MPI_COMM_NULL)
      MPI_Comm_free(&dump_comm);
  #endif
}

void SimulationData::setCells(const int nx, const int ny, const int nz)
{
  if (   nx % (nprocsx * FluidBlock::sizeX) != 0
      || ny % (nprocsy * FluidBlock::sizeY) != 0
      || nz % (nprocsz * FluidBlock::sizeZ) != 0) {
    throw std::invalid_argument("Number of cells must be multiple of "
                                "block size * number of processes.");
  }
  bpdx = nx / FluidBlock::sizeX;
  bpdy = ny / FluidBlock::sizeY;
  bpdz = nz / FluidBlock::sizeZ;
}

void SimulationData::startProfiler(std::string name) const
{
  profiler->push_start(name);
}
void SimulationData::stopProfiler() const
{
  profiler->pop_stop();
}
void SimulationData::printResetProfiler()
{
  profiler->printSummary();
  profiler->reset();
}

CubismUP_3D_NAMESPACE_END
