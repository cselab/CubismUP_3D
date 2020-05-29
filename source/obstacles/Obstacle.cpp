//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Obstacle.h"
#include "../utils/BufferedLogger.h"

#include <Cubism/ArgumentParser.h>
#include <gsl/gsl_linalg.h>
#include <fstream>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

using UDEFMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][3];
using CHIMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
static constexpr double DBLEPS = std::numeric_limits<double>::epsilon();

ObstacleArguments::ObstacleArguments(
        const SimulationData & sim,
        ArgumentParser &parser)
{
  length = parser("-L").asDouble();          // Mandatory.
  position[0] = parser("-xpos").asDouble();  // Mandatory.
  position[1] = parser("-ypos").asDouble(sim.extent[1] / 2);
  position[2] = parser("-zpos").asDouble(sim.extent[2] / 2);
  quaternion[0] = parser("-quat0").asDouble(0.0);
  quaternion[1] = parser("-quat1").asDouble(0.0);
  quaternion[2] = parser("-quat2").asDouble(0.0);
  quaternion[3] = parser("-quat3").asDouble(0.0);
  planarAngle = parser("-planarAngle").asDouble(0.0) / 180 * M_PI;
  const double q_length = std::sqrt(quaternion[0]*quaternion[0]
                                 +  quaternion[1]*quaternion[1]
                                 +  quaternion[2]*quaternion[2]
                                 +  quaternion[3]*quaternion[3]);

  if(std::fabs(q_length-1.0) > 5*EPS) {
    quaternion[0] = std::cos(0.5*planarAngle);
    quaternion[1] = 0;
    quaternion[2] = 0;
    quaternion[3] = std::sin(0.5*planarAngle);
  } else {
    if(std::fabs(planarAngle) > 0 && sim.rank == 0)
      printf("WARNING: Obstacle arguments include both quaternions and "
             "planarAngle. Quaterion arguments have priority and therefore "
             "planarAngle will be ignored.");

    planarAngle = 2 * std::atan2(quaternion[3], quaternion[0]);
  }

  // if true, obstacle will never change its velocity:
  // bForcedInLabFrame = parser("-bForcedInLabFrame").asBool(false);
  bool bFSM_alldir = parser("-bForcedInSimFrame").asBool(false);
  bForcedInSimFrame[0] = bFSM_alldir || parser("-bForcedInSimFrame_x").asBool(false);
  bForcedInSimFrame[1] = bFSM_alldir || parser("-bForcedInSimFrame_y").asBool(false);
  bForcedInSimFrame[2] = bFSM_alldir || parser("-bForcedInSimFrame_z").asBool(false);

  // only active if corresponding bForcedInLabFrame is true:
  enforcedVelocity[0] = -parser("-xvel").asDouble(0.0);
  enforcedVelocity[1] = -parser("-yvel").asDouble(0.0);
  enforcedVelocity[2] = -parser("-zvel").asDouble(0.0);

  bFixToPlanar = parser("-bFixToPlanar").asBool(false);

  // this is different, obstacle can change the velocity, but sim frame will follow:
  bool bFOR_alldir = parser("-bFixFrameOfRef").asBool(false);
  bFixFrameOfRef[0] = bFOR_alldir || parser("-bFixFrameOfRef_x").asBool(false);
  bFixFrameOfRef[1] = bFOR_alldir || parser("-bFixFrameOfRef_y").asBool(false);
  bFixFrameOfRef[2] = bFOR_alldir || parser("-bFixFrameOfRef_z").asBool(false);

  // To force forced obst. into computeForces or to force self-propelled
  // into diagnostics forces (tasso del tasso del tasso):
  // If untouched forced only do diagnostics and selfprop only do surface.
  bComputeForces = parser("-computeForces").asBool(false);
}

Obstacle::Obstacle(SimulationData&s, ArgumentParser&p)
    : Obstacle( s, ObstacleArguments(s, p) ) { }

Obstacle::Obstacle(
    SimulationData& s, const ObstacleArguments &args)
    : Obstacle(s)
{
  length = args.length;
  position[0] = args.position[0];
  position[1] = args.position[1];
  position[2] = args.position[2];
  absPos[0] = position[0]; absPos[1] = position[1]; absPos[2] = position[2];
  quaternion[0] = args.quaternion[0];
  quaternion[1] = args.quaternion[1];
  quaternion[2] = args.quaternion[2];
  quaternion[3] = args.quaternion[3];
  _2Dangle = args.planarAngle;

  if (!sim.rank) {
    printf("Obstacle L=%g, pos=[%g %g %g], q=[%g %g %g %g]\n",
           length, position[0], position[1], position[2],
           quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  }

  const double one = std::sqrt(
          quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);

  if (std::fabs(one - 1.0) > 5 * DBLEPS) {
    printf("Parsed quaternion length is not equal to one. It really ought to be.\n");
    fflush(0);
    abort();
  }
  if (length < 5 * EPS) {
    printf("Parsed length is equal to zero. It really ought not to be.\n");
    fflush(0);
    abort();
  }

  for (int d = 0; d < 3; ++d) {
    bForcedInSimFrame[d] = args.bForcedInSimFrame[d];
    if (bForcedInSimFrame[d]) {
      transVel_imposed[d] = transVel[d] = args.enforcedVelocity[d];
      if (!sim.rank) {
         printf("Obstacle forced to move relative to sim domain with constant %c-vel: %f\n",
                "xyz"[d], transVel[d]);
      }
    }
  }

  const bool anyVelForced = bForcedInSimFrame[0] || bForcedInSimFrame[1] || bForcedInSimFrame[2];
  if(anyVelForced) {
    if (!sim.rank) printf("Obstacle has no angular velocity.\n");
    bBlockRotation[0] = true;
    bBlockRotation[1] = true;
    bBlockRotation[2] = true;
  }
  const bool bFixToPlanar = args.bFixToPlanar;
  if(bFixToPlanar) {
    if (!sim.rank) printf("Obstacle motion restricted to constant Z-plane.\n");
    bForcedInSimFrame[2] = true;
    transVel_imposed[2] = 0;
    //bBlockRotation[2] = true;
    bBlockRotation[1] = true;
    bBlockRotation[0] = true;
  }

  bFixFrameOfRef[0] = args.bFixFrameOfRef[0];
  bFixFrameOfRef[1] = args.bFixFrameOfRef[1];
  bFixFrameOfRef[2] = args.bFixFrameOfRef[2];
  bForces = args.bComputeForces;
}

void Obstacle::computeVelocities()
{
  double A[6][6] = {
 {     penalM,         0.0,         0.0,         0.0, +penalCM[2], -penalCM[1]},
 {        0.0,      penalM,         0.0, -penalCM[2],         0.0, +penalCM[0]},
 {        0.0,         0.0,      penalM, +penalCM[1], -penalCM[0],         0.0},
 {        0.0, -penalCM[2], +penalCM[1],   penalJ[0],   penalJ[3],   penalJ[4]},
 {+penalCM[2],         0.0, -penalCM[0],   penalJ[3],   penalJ[1],   penalJ[5]},
 {-penalCM[1], +penalCM[0],         0.0,   penalJ[4],   penalJ[5],   penalJ[2]}
  };

  // TODO here we can add dt * appliedForce/Torque[i]
  double b[6] = {
    penalLmom[0], penalLmom[1], penalLmom[2],
    penalAmom[0], penalAmom[1], penalAmom[2]
  };

  //Momenta are conserved if a dof (a row of mat A) is not externally forced
  //This means that if obstacle is free to move according to fluid forces,
  //momenta after penal should be equal to moments before penal!
  //If dof is forced, change in momt. assumed to be entirely due to forcing.
  //In this case, leave row diagonal to compute change in momt for post/dbg.
  //If dof (row) is free then i need to fill the non-diagonal terms.
  if( bForcedInSimFrame[0] ) { //then momenta not conserved in this dof
    A[0][1] = 0; A[0][2] = 0; A[0][3] = 0; A[0][4] = 0; A[0][5] = 0;
    b[0] = penalM * transVel_imposed[0]; // multply by penalM for conditioning
  }
  if( bForcedInSimFrame[1] ) { //then momenta not conserved in this dof
    A[1][0] = 0; A[1][2] = 0; A[1][3] = 0; A[1][4] = 0; A[1][5] = 0;
    b[1] = penalM * transVel_imposed[1];
  }
  if( bForcedInSimFrame[2] ) { //then momenta not conserved in this dof
    A[2][0] = 0; A[2][1] = 0; A[2][3] = 0; A[2][4] = 0; A[2][5] = 0;
    b[2] = penalM * transVel_imposed[2];
  }
  if( bBlockRotation[0] ) { //then momenta not conserved in this dof
    A[3][0] = 0; A[3][1] = 0; A[3][2] = 0; A[3][4] = 0; A[3][5] = 0;
    b[3] = 0; // TODO IMPOSED ANG VEL?
  }
  if( bBlockRotation[1] ) { //then momenta not conserved in this dof
    A[4][0] = 0; A[4][1] = 0; A[4][2] = 0; A[4][3] = 0; A[4][5] = 0;
    b[4] = 0; // TODO IMPOSED ANG VEL?
  }
  if( bBlockRotation[2] ) { //then momenta not conserved in this dof
    A[5][0] = 0; A[5][1] = 0; A[5][2] = 0; A[5][3] = 0; A[5][4] = 0;
    b[5] = 0; // TODO IMPOSED ANG VEL?
  }

  gsl_matrix_view Agsl = gsl_matrix_view_array (&A[0][0], 6, 6);
  gsl_vector_view bgsl = gsl_vector_view_array (b, 6);
  gsl_vector *xgsl = gsl_vector_alloc (6);
  int sgsl;
  gsl_permutation * permgsl = gsl_permutation_alloc (6);
  gsl_linalg_LU_decomp (& Agsl.matrix, permgsl, & sgsl);
  gsl_linalg_LU_solve (& Agsl.matrix, permgsl, & bgsl.vector, xgsl);
  transVel_computed[0] = gsl_vector_get(xgsl, 0);
  transVel_computed[1] = gsl_vector_get(xgsl, 1);
  transVel_computed[2] = gsl_vector_get(xgsl, 2);
  angVel_computed[0]   = gsl_vector_get(xgsl, 3);
  angVel_computed[1]   = gsl_vector_get(xgsl, 4);
  angVel_computed[2]   = gsl_vector_get(xgsl, 5);

  gsl_permutation_free (permgsl);
  gsl_vector_free (xgsl);
  //if(sim.verbose)
  //{
  //  printf("um:%e lm:%e am:%e m:%e j:%e u:%e v:%e a:%e\n",
  //  penalLmom[0], penalLmom[1], penalAmom[2], penalM, penalJ[2],
  //  transVel_computed[0], transVel_computed[1], angVel_computed[2]);
  //}
  force[0] = mass * (transVel_computed[0] - transVel[0]) / sim.dt;
  force[1] = mass * (transVel_computed[1] - transVel[1]) / sim.dt;
  force[2] = mass * (transVel_computed[2] - transVel[2]) / sim.dt;
  const std::array<double,3> dAv = {
    (angVel_computed[0] - angVel[0]) / sim.dt,
    (angVel_computed[1] - angVel[1]) / sim.dt,
    (angVel_computed[2] - angVel[2]) / sim.dt
  };
  torque[0] = J[0] * dAv[0] + J[3] * dAv[1] + J[4] * dAv[2];
  torque[1] = J[3] * dAv[0] + J[1] * dAv[1] + J[5] * dAv[2];
  torque[2] = J[4] * dAv[0] + J[5] * dAv[1] + J[2] * dAv[2];

  if(bForcedInSimFrame[0]) {
    assert( std::fabs(transVel[0] - transVel_imposed[0]) < 1e-12 );
    transVel[0] = transVel_imposed[0];
  } else transVel[0] = transVel_computed[0];

  if(bForcedInSimFrame[1]) {
    assert( std::fabs(transVel[1] - transVel_imposed[1]) < 1e-12 );
    transVel[1] = transVel_imposed[1];
  } else transVel[1] = transVel_computed[1];

  if(bForcedInSimFrame[2]) {
    assert( std::fabs(transVel[2] - transVel_imposed[2]) < 1e-12 );
    transVel[2] = transVel_imposed[2];
  } else transVel[2] = transVel_computed[2];

  if( bBlockRotation[0] ) {
    assert( std::fabs(angVel[0] - 0) < 1e-12 );
    angVel[0] = 0;
  } else angVel[0] = angVel_computed[0];

  if( bBlockRotation[1] ) {
    assert( std::fabs(angVel[1] - 0) < 1e-12 );
    angVel[1] = 0;
  } else angVel[1] = angVel_computed[1];

  if( bBlockRotation[2] ) {
    assert( std::fabs(angVel[2] - 0) < 1e-12 );
    angVel[2] = 0;
  } else angVel[2] = angVel_computed[2];
}

void Obstacle::computeForces()
{
  static const int nQoI = ObstacleBlock::nQoI;
  std::vector<double> sum = std::vector<double>(nQoI, 0);
  for (auto & block : obstacleBlocks) {
    if(block == nullptr) continue;
    block->sumQoI(sum);
  }

  MPI_Allreduce(MPI_IN_PLACE, sum.data(), nQoI, MPI_DOUBLE, MPI_SUM, grid->getCartComm());

  //additive quantities: (check against order in sumQoI of ObstacleBlocks.h )
  unsigned k = 0;
  surfForce[0]  = sum[k++]; surfForce[1]  = sum[k++]; surfForce[2]  = sum[k++];
  presForce[0]  = sum[k++]; presForce[1]  = sum[k++]; presForce[2]  = sum[k++];
  viscForce[0]  = sum[k++]; viscForce[1]  = sum[k++]; viscForce[2]  = sum[k++];
  surfTorque[0] = sum[k++]; surfTorque[1] = sum[k++]; surfTorque[2] = sum[k++];
  gamma[0]      = sum[k++]; gamma[1]      = sum[k++]; gamma[2]      = sum[k++];
  drag          = sum[k++]; thrust        = sum[k++]; Pout          = sum[k++];
  PoutBnd       = sum[k++]; defPower      = sum[k++]; defPowerBnd   = sum[k++];
  pLocom        = sum[k++];

  const double vel_norm = std::sqrt(transVel[0]*transVel[0]
                                  + transVel[1]*transVel[1]
                                  + transVel[2]*transVel[2]);
  //derived quantities:
  Pthrust    = thrust*vel_norm;
  Pdrag      =   drag*vel_norm;
  EffPDef    = Pthrust/(Pthrust-std::min(defPower,(double)0)+EPS);
  EffPDefBnd = Pthrust/(Pthrust-         defPowerBnd        +EPS);

  #if defined(CUP_DUMP_SURFACE_BINARY) && !defined(RL_LAYER)
  if (sim.bDump) {
    char buf[500];
    sprintf(buf,"surface_%02d_%07d_rank%03d.raw",obstacleID,sim.step,sim.rank);
    FILE * pFile = fopen (buf, "wb");
    for(auto & block : obstacleBlocks) {
      if(block == nullptr) continue;
      block->print(pFile);
    }
    fflush(pFile);
    fclose(pFile);
  }
  #endif
  _writeSurfForcesToFile();
  _writeDiagForcesToFile();
}

void Obstacle::update()
{
  const Real dt = sim.dt;
  position[0] += dt * ( transVel[0] + sim.uinf[0] );
  position[1] += dt * ( transVel[1] + sim.uinf[1] );
  position[2] += dt * ( transVel[2] + sim.uinf[2] );
  absPos[0] += dt * transVel[0];
  absPos[1] += dt * transVel[1];
  absPos[2] += dt * transVel[2];
  const double Q[] = {quaternion[0],quaternion[1],quaternion[2],quaternion[3]};
  const double dqdt[4] = {
    .5*( - angVel[0]*Q[1] - angVel[1]*Q[2] - angVel[2]*Q[3] ),
    .5*( + angVel[0]*Q[0] + angVel[1]*Q[3] - angVel[2]*Q[2] ),
    .5*( - angVel[0]*Q[3] + angVel[1]*Q[0] + angVel[2]*Q[1] ),
    .5*( + angVel[0]*Q[2] - angVel[1]*Q[1] + angVel[2]*Q[0] )
  };

  // normality preserving advection (Simulation of colliding constrained rigid bodies - Kleppmann 2007 Cambridge University, p51)
  // move the correct distance on the quaternion unit ball surface, end up with normalized quaternion
  const double DQ[4] = { dqdt[0]*dt, dqdt[1]*dt, dqdt[2]*dt, dqdt[3]*dt };
  const double DQn = std::sqrt(DQ[0]*DQ[0]+DQ[1]*DQ[1]+DQ[2]*DQ[2]+DQ[3]*DQ[3]);

  if(DQn>DBLEPS)
  {
    const double tanF = std::tan(DQn)/DQn;
    const double D[4] = {
      Q[0] +tanF*DQ[0], Q[1] +tanF*DQ[1], Q[2] +tanF*DQ[2], Q[3] +tanF*DQ[3],
    };
    const double invD = 1/std::sqrt(D[0]*D[0]+D[1]*D[1]+D[2]*D[2]+D[3]*D[3]);
    quaternion[0] = D[0] * invD; quaternion[1] = D[1] * invD;
    quaternion[2] = D[2] * invD; quaternion[3] = D[3] * invD;
  }

  //_2Dangle += dt*angVel[2];
  const double old2DA = _2Dangle;
  //keep consistency: get 2d angle from quaternions:
  _2Dangle = 2*std::atan2(quaternion[3], quaternion[0]);
  const double err = std::fabs(_2Dangle-old2DA-dt*angVel[2]);
  if(err>EPS && !sim.rank)
    printf("Discrepancy in angvel from quaternions: %f (%f %f)\n",
      err, (_2Dangle-old2DA)/dt, angVel[2]);

  #ifndef NDEBUG
  if(sim.rank==0)
  {
    #ifdef CUP_VERBOSE
     printf("POSITION INFO AFTER UPDATE T, DT: %lf %lf\n", sim.time, sim.dt);
     printf("POS: %lf %lf %lf\n", position[0], position[1], position[2]);
     printf("TVL: %lf %lf %lf\n", transVel[0], transVel[1], transVel[2]);
     printf("QUT: %lf %lf %lf %lf\n", quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
     printf("AVL: %lf %lf %lf\n", angVel[0], angVel[1], angVel[2]);
     fflush(stdout);
    #endif
  }
  const double q_length=std::sqrt(quaternion[0]*quaternion[0]
        +  quaternion[1]*quaternion[1]
        +  quaternion[2]*quaternion[2]
        +  quaternion[3]*quaternion[3]);
  assert(std::abs(q_length-1.0) < 5*EPS);
  #endif

  if(dt>0) _writeComputedVelToFile();
}

void Obstacle::create()
{
  printf("Entered the wrong create operator\n");
  fflush(0); exit(1);
}

void Obstacle::finalize()
{ }

std::array<double,3> Obstacle::getTranslationVelocity() const
{
  return std::array<double,3> {{transVel[0],transVel[1],transVel[2]}};
}

std::array<double,3> Obstacle::getAngularVelocity() const
{
  return std::array<double,3> {{angVel[0],angVel[1],angVel[2]}};
}

std::array<double,3> Obstacle::getCenterOfMass() const
{
  return std::array<double,3> {{centerOfMass[0],centerOfMass[1],centerOfMass[2]}};
}

void Obstacle::save(std::string filename)
{
  if(sim.rank!=0) return;
  #ifdef RL_LAYER
  sr.save(sim.step,filename);
  #endif
  std::ofstream savestream;
  savestream.setf(std::ios::scientific);
  savestream.precision(std::numeric_limits<Real>::digits10 + 1);
  savestream.open(filename+".txt");
  if (!savestream) {
    fprintf(stderr, "Couldn't open \"%s.txt\".\n", filename.c_str());
    fflush(0); exit(1);
  }
  savestream<<sim.time<<std::endl;
  savestream<<position[0]<<"\t"<<position[1]<<"\t"<<position[2]<<std::endl;
  savestream<<absPos[0]<<"\t"<<absPos[1]<<"\t"<<absPos[2]<<std::endl;
  savestream<<quaternion[0]<<"\t"<<quaternion[1]<<"\t"<<quaternion[2]<<"\t"<<quaternion[3]<<std::endl;
  savestream<<transVel[0]<<"\t"<<transVel[1]<<"\t"<<transVel[2]<<std::endl;
  savestream<<angVel[0]<<"\t"<<angVel[1]<<"\t"<<angVel[2]<<std::endl;
  savestream<<_2Dangle<<std::endl;
}

void Obstacle::restart(std::string filename)
{
  #ifdef RL_LAYER
    sr.restart(filename);
  #endif
  std::ifstream restartstream;
  restartstream.open(filename+".txt");
  if(!restartstream.good()){
    printf("Could not restart from file\n");
    return;
  }
  Real restart_time;
  restartstream >> restart_time;
  //assert(std::abs(restart_time-t) < 1e-9);

  restartstream>>position[0]>>position[1]>>position[2];
  restartstream>>absPos[0]>>absPos[1]>>absPos[2];
  restartstream>>quaternion[0]>>quaternion[1]>>quaternion[2]>>quaternion[3];
  restartstream>>transVel[0]>>transVel[1]>>transVel[2];
  restartstream>>angVel[0]>>angVel[1]>>angVel[2];
  restartstream >> _2Dangle;
  restartstream.close();

  {
  printf("RESTARTED BODY:\n");
  printf("TIME: \t%lf\n", restart_time);
  printf("POS:  \t%lf %lf %lf\n", position[0], position[1], position[2]);
  printf("ABS POS: \t%lf %lf %lf\n", absPos[0], absPos[1], absPos[2]);
  printf("ANGLE:\t%lf %lf %lf %lf\n", quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  printf("TVEL: \t%lf %lf %lf\n", transVel[0], transVel[1], transVel[2]);
  printf("AVEL: \t%lf %lf %lf\n", angVel[0], angVel[1], angVel[2]);
  printf("2D angle: \t%lf\n", _2Dangle);
  fflush(stdout);
  }
}

void Obstacle::Accept(ObstacleVisitor * visitor)
{
 visitor->visit(this);
}

#ifdef RL_LAYER
void Obstacle::getSkinsAndPOV(Real& x, Real& y, Real& th,
  Real*& pXL, Real*& pYL, Real*& pXU, Real*& pYU, int& Npts)
{
  printf("Entered the wrong get skin operator\n");
  fflush(0);
  abort();
}

void Obstacle::execute(const int iAgent, const double time, const std::vector<double> action)
{
  printf("Entered the wrong execute operator\n");
  fflush(0);
  abort();
}

void Obstacle::interpolateOnSkin(const double time, const int stepID, bool dumpWake)
{
  //printf("Entered the wrong interpolate operator\n");
  //fflush(0);
  //abort();
}

#endif

void Obstacle::_writeComputedVelToFile()
{
  if(sim.rank!=0) return;
  std::stringstream ssR;
  ssR<<"computedVelocity_"<<obstacleID<<".dat";
  std::stringstream &savestream = logger.get_stream(ssR.str());
  const std::string tab("\t");

  if(sim.step==0 && not printedHeaderVels) {
    printedHeaderVels = true;
    savestream<<"step"<<tab<<"time"<<tab<<"CMx"<<tab<<"CMy"<<tab<<"CMz"<<tab
    <<"quat_0"<<tab<<"quat_1"<<tab<<"quat_2"<<tab<<"quat_3"<<tab
    <<"vel_x"<<tab<<"vel_y"<<tab<<"vel_z"<<tab
    <<"angvel_x"<<tab<<"angvel_y"<<tab<<"angvel_z"<<tab<<"mass"<<tab
    <<"J0"<<tab<<"J1"<<tab<<"J2"<<tab<<"J3"<<tab<<"J4"<<tab<<"J5"<<std::endl;
  }

  savestream<<sim.step<<tab;
  savestream.setf(std::ios::scientific);
  savestream.precision(std::numeric_limits<float>::digits10 + 1);
  savestream <<sim.time<<tab<<absPos[0]<<tab<<absPos[1]<<tab<<absPos[2]<<tab
    <<quaternion[0]<<tab<<quaternion[1]<<tab<<quaternion[2]<<tab<<quaternion[3]
    <<tab<<transVel[0]<<tab<<transVel[1]<<tab<<transVel[2]
    <<tab<<angVel[0]<<tab<<angVel[1]<<tab<<angVel[2]<<tab<<mass<<tab
    <<J[0]<<tab<<J[1]<<tab<<J[2]<<tab<<J[3]<<tab<<J[4]<<tab<<J[5]<<std::endl;
}

void Obstacle::_writeSurfForcesToFile()
{
  if(sim.rank!=0) return;
  std::stringstream fnameF, fnameP;
  fnameF<<"forceValues_"<<(!isSelfPropelled?"surface_":"")<<obstacleID<<".dat";
  std::stringstream &ssF = logger.get_stream(fnameF.str());
  const std::string tab("\t");
  if(sim.step==0) {
    ssF<<"step"<<tab<<"time"<<tab<<"mass"<<tab<<"force_x"<<tab<<"force_y"
    <<tab<<"force_z"<<tab<<"torque_x"<<tab<<"torque_y"<<tab<<"torque_z"
    <<tab<<"presF_x"<<tab<<"presF_y"<<tab<<"presF_z"<<tab<<"viscF_x"
    <<tab<<"viscF_y"<<tab<<"viscF_z"<<tab<<"gamma_x"<<tab<<"gamma_y"
    <<tab<<"gamma_z"<<tab<<"drag"<<tab<<"thrust"<<std::endl;
  }

  ssF << sim.step << tab;
  ssF.setf(std::ios::scientific);
  ssF.precision(std::numeric_limits<float>::digits10 + 1);
  ssF<<sim.time<<tab<<mass<<tab<<surfForce[0]<<tab<<surfForce[1]<<tab<<surfForce[2]
     <<tab<<surfTorque[0]<<tab<<surfTorque[1]<<tab<<surfTorque[2]<<tab
     <<presForce[0]<<tab<<presForce[1]<<tab<<presForce[2]<<tab<<viscForce[0]
     <<tab<<viscForce[1]<<tab<<viscForce[2]<<tab<<gamma[0]<<tab<<gamma[1]
     <<tab<<gamma[2]<<tab<<drag<<tab<<thrust<<std::endl;

  fnameP<<"powerValues_"<<(!isSelfPropelled?"surface_":"")<<obstacleID<<".dat";
  std::stringstream &ssP = logger.get_stream(fnameP.str());
  if(sim.step==0) {
    ssP<<"step"<<tab<<"time"<<tab<<"Pthrust"<<tab<<"Pdrag"<<tab
       <<"Pout"<<tab<<"pDef"<<tab<<"etaPDef"<<tab<<"pLocom"<<tab
       <<"PoutBnd"<<tab<<"defPowerBnd"<<tab<<"etaPDefBnd"<<std::endl;
  }
  ssP << sim.step << tab;
  ssP.setf(std::ios::scientific);
  ssP.precision(std::numeric_limits<float>::digits10 + 1);
  // Output defpowers to text file with the correct sign
  ssP<<sim.time<<tab<<Pthrust<<tab<<Pdrag<<tab<<Pout<<tab<<-defPower<<tab<<EffPDef
     <<tab<<pLocom<<tab<<PoutBnd<<tab<<-defPowerBnd<<tab<<EffPDefBnd<<std::endl;
}

void Obstacle::_writeDiagForcesToFile()
{
  if(sim.rank!=0) return;
  std::stringstream fnameF;
  fnameF<<"forceValues_"<<(isSelfPropelled?"penalization_":"")<<obstacleID<<".dat";
  std::stringstream &ssF = logger.get_stream(fnameF.str());
  const std::string tab("\t");
  if(sim.step==0) {
    ssF << "step" << tab << "time" << tab << "mass" << tab
    << "force_x" << tab << "force_y" << tab << "force_z" << tab
    << "torque_x" << tab << "torque_y" << tab << "torque_z"<< tab
    << "penalLmom_x" << tab << "penalLmom_y" << tab << "penalLmom_z" << tab
    << "penalAmom_x" << tab << "penalAmom_y" << tab << "penalAmom_z" << tab
    << "penalCM_x" << tab << "penalCM_y" << tab << "penalCM_z" << tab
   << "linVel_comp_x" << tab << "linVel_comp_y" << tab << "linVel_comp_z" << tab
   << "angVel_comp_x" << tab << "angVel_comp_y" << tab << "angVel_comp_z" << tab
   << "penalM"<<tab << "penalJ0" << tab << "penalJ1" << tab << "penalJ2" << tab
   << "penalJ3" << tab << "penalJ4" << tab << "penalJ5" << std::endl;
  }

  ssF << sim.step << tab;
  ssF.setf(std::ios::scientific);
  ssF.precision(std::numeric_limits<float>::digits10 + 1);
  ssF<<sim.time<<tab<<mass<<tab<<force[0]<<tab<<force[1]<<tab<<force[2]<<tab
     <<torque[0]<<tab<<torque[1]<<tab<<torque[2]
     <<tab<<penalLmom[0]<<tab<<penalLmom[1]<<tab<<penalLmom[2]
     <<tab<<penalAmom[0]<<tab<<penalAmom[1]<<tab<<penalAmom[2]
     <<tab<<penalCM[0]<<tab<<penalCM[1]<<tab<<penalCM[2]
     <<tab<<transVel_computed[0]<<tab<<transVel_computed[1]<<tab<<transVel_computed[2]
     <<tab<<angVel_computed[0]<<tab<<angVel_computed[1]<<tab<<angVel_computed[2]
     <<tab<<penalM<<tab<<penalJ[0]<<tab<<penalJ[1]<<tab<<penalJ[2]
     <<tab<<penalJ[3]<<tab<<penalJ[4]<<tab<<penalJ[5] <<std::endl;
}

CubismUP_3D_NAMESPACE_END
