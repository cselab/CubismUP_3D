//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#include "Fish.h"
#include "FishLibrary.h"

#include <Cubism/ArgumentParser.h>
#include <Cubism/HDF5Dumper_MPI.h>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

Fish::Fish(SimulationData&s, ArgumentParser&p) : Obstacle(s, p)
{
  p.unset_strict_mode();
  Tperiod = p("-T").asDouble(1.0);
  phaseShift = p("-phi").asDouble(0.0);

  //PID knobs
  bCorrectTrajectory = p("-Correct").asBool(false);

  // Main amplitude modulation for all fish should be amplitudeFactor, otherwise
  // new fish classes should take care to handle isSelfPropelled correctly.
  // Additional shaping of the gait (eg curvatures, carling/quadratic factor)
  // is then multiplied by this arg. If amplitudeFactor=0 fish is assumed towed.
  if(p("-amplitudeFactor").asDouble(1.0)>0)
    isSelfPropelled = true;

  bCorrectPosition = p("-bCorrectPosition").asBool(false);
  const double hh = 0.5*sim.maxH();
  position[2] = p("-zpos").asDouble(sim.extent[2]/2 + hh);

  bHasSkin = true;
}

Fish::~Fish()
{
  if(myFish not_eq nullptr) delete myFish;
}

void Fish::integrateMidline()
{
  volume_internal = myFish->integrateLinearMomentum(CoM_internal, vCoM_internal);
  assert(volume_internal > std::numeric_limits<Real>::epsilon());
  myFish->changeToCoMFrameLinear(CoM_internal, vCoM_internal);
  // compute angular velocity resulting from undulatory deformation
  if(sim.dt>0) angvel_internal_prev = angvel_internal;
  myFish->integrateAngularMomentum(angvel_internal);
  // because deformations cannot impose rotation, we both subtract the angvel
  // from the midline velocity and remove it from the internal angle:
  theta_internal -= sim.dt * (angvel_internal + angvel_internal_prev)/2;
  J_internal = myFish->J;
  myFish->changeToCoMFrameAngular(theta_internal, angvel_internal);

  #ifndef NDEBUG
  {
    double dummy_CoM_internal[2], dummy_vCoM_internal[2], dummy_angvel_internal;
    // check that things are zero
    const double volume_internal_check = myFish->integrateLinearMomentum(dummy_CoM_internal,dummy_vCoM_internal);
    myFish->integrateAngularMomentum(dummy_angvel_internal);
    const double EPS = 10*std::numeric_limits<Real>::epsilon();
    assert(std::fabs(dummy_CoM_internal[0])<EPS);
    assert(std::fabs(dummy_CoM_internal[1])<EPS);
    assert(std::fabs(myFish->linMom[0])<EPS);
    assert(std::fabs(myFish->linMom[1])<EPS);
    assert(std::fabs(myFish->angMom)<EPS);
    assert(std::fabs(volume_internal - volume_internal_check) < EPS);
  }
  #endif
  //MPI_Barrier(grid->getCartComm());
  myFish->surfaceToCOMFrame(theta_internal,CoM_internal);
}

std::vector<VolumeSegment_OBB> Fish::prepare_vSegments()
{
  /*
    - VolumeSegment_OBB's volume cannot be zero
    - therefore no VolumeSegment_OBB can be only occupied by extension midline
      points (which have width and height = 0)
    - performance of create seems to decrease if VolumeSegment_OBB are bigger
    - this is the smallest number of VolumeSegment_OBB (Nsegments) and points in
      the midline (Nm) to ensure at least one non ext. point inside all segments
   */
  const int Nsegments = std::ceil((myFish->Nm-1.)/8);
  const int Nm = myFish->Nm;
  assert((Nm-1)%Nsegments==0);

  std::vector<VolumeSegment_OBB> vSegments(Nsegments);
  #pragma omp parallel for schedule(static)
  for(int i=0; i<Nsegments; ++i)
  {
    const int nextidx = (i+1)*(Nm-1)/Nsegments;
    const int idx = i * (Nm-1)/Nsegments;
    // find bounding box based on this
    Real bbox[3][2] = {{1e9, -1e9}, {1e9, -1e9}, {1e9, -1e9}};
    for(int ss=idx; ss<=nextidx; ++ss)
    {
      const Real xBnd[2] = {myFish->rX[ss] - myFish->norX[ss]*myFish->width[ss],
          myFish->rX[ss] + myFish->norX[ss]*myFish->width[ss]};
      const Real yBnd[2] = {myFish->rY[ss] - myFish->norY[ss]*myFish->width[ss],
          myFish->rY[ss] + myFish->norY[ss]*myFish->width[ss]};
      const Real zBnd[2] = {-myFish->height[ss], +myFish->height[ss]};
      const Real maxX=std::max(xBnd[0],xBnd[1]), minX=std::min(xBnd[0],xBnd[1]);
      const Real maxY=std::max(yBnd[0],yBnd[1]), minY=std::min(yBnd[0],yBnd[1]);
      const Real maxZ=std::max(zBnd[0],zBnd[1]), minZ=std::min(zBnd[0],zBnd[1]);
      bbox[0][0] = std::min(bbox[0][0], minX);
      bbox[0][1] = std::max(bbox[0][1], maxX);
      bbox[1][0] = std::min(bbox[1][0], minY);
      bbox[1][1] = std::max(bbox[1][1], maxY);
      bbox[2][0] = std::min(bbox[2][0], minZ);
      bbox[2][1] = std::max(bbox[2][1], maxZ);
    }

    vSegments[i].prepare(std::make_pair(idx,nextidx), bbox, sim.maxH());
    vSegments[i].changeToComputationalFrame(position,quaternion);
  }
  return vSegments;
}

using intersect_t = std::vector<std::vector<VolumeSegment_OBB*>>;
intersect_t Fish::prepare_segPerBlock(vecsegm_t& vSegments)
{
  const std::vector<cubism::BlockInfo>& vInfo = sim.vInfo();
  std::vector<std::vector<VolumeSegment_OBB*>> ret(vInfo.size());

  // clear deformation velocities
  for(auto & entry : obstacleBlocks) {
    if(entry == nullptr) continue;
    delete entry;
    entry = nullptr;
  }
  obstacleBlocks.resize(vInfo.size(), nullptr);

  #pragma omp parallel for schedule(dynamic, 1)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    const BlockInfo & info = vInfo[i];
    const FluidBlock & b = *(FluidBlock*)info.ptrBlock;

    for(size_t s=0; s<vSegments.size(); ++s)
      if(vSegments[s].isIntersectingWithAABB(b.min_pos.data(), b.max_pos.data())) {
        VolumeSegment_OBB*const ptr  = & vSegments[s];
        ret[info.blockID].push_back( ptr );
      }

    // allocate new blocks if necessary
    if( ret[info.blockID].size() > 0 ) {
      assert( obstacleBlocks[info.blockID] == nullptr );
      ObstacleBlock * const block = new ObstacleBlock();
      assert(block not_eq nullptr);
      obstacleBlocks[info.blockID] = block;
      block->clear();
    }
  }
  return ret;
}

void Fish::writeSDFOnBlocks(const intersect_t& segmentsPerBlock)
{
  const std::vector<cubism::BlockInfo>& vInfo = sim.vInfo();
  #pragma omp parallel
  {
    PutFishOnBlocks putfish(myFish, position, quaternion);

    #pragma omp for schedule(dynamic, 1)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const BlockInfo info = vInfo[i];
      const std::vector<VolumeSegment_OBB*>& S = segmentsPerBlock[info.blockID];
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      if(S.size() > 0)
      {
        assert(obstacleBlocks[info.blockID] not_eq nullptr);
        ObstacleBlock*const block = obstacleBlocks[info.blockID];
        putfish(info, b, block, S);
      }
      else assert(obstacleBlocks[info.blockID] == nullptr);
    }
  }

  #if 0
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < (int)vInfo.size(); ++i) {
      const BlockInfo info = vInfo[i];
      const auto pos = obstacleBlocks[info.blockID];
      if(pos == nullptr) continue;
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        //b(ix,iy,iz).chi = pos->second->chi[iz][iy][ix];//b(ix,iy,iz).tmpU;
        b(ix,iy,iz).u = b(ix,iy,iz).tmpU;
        b(ix,iy,iz).v = b(ix,iy,iz).tmpV;
        b(ix,iy,iz).w = b(ix,iy,iz).tmpW;
      }
    }
  }
  DumpHDF5_MPI<StreamerVelocityVector, DumpReal>(*grid, 0, 0, "SFD", "./");
  abort();
  #endif
}

void Fish::create()
{
  // STRATEGY
  // we need some things already
  // - the internal angle at the previous timestep, obtained from integrating the actual def velocities
  // (not the imposed deformation velocies, because they dont have zero ang mom)
  // - the internal angular velocity at previous timestep

  // 1. create midline
  // 2. integrate to find CoM, angular velocity, etc
  // 3. shift midline to CoM frame: zero internal linear momentum and angular momentum

  // 4. split the fish into segments (according to s)
  // 5. rotate the segments to computational frame (comp CoM and angle)
  // 6. for each Block in the domain, find those segments that intersect it
  // 7. for each of those blocks, allocate an ObstacleBlock

  // 8. put the 3D shape on the grid: SDF-P2M for sdf, normal P2M for udef
  //apply_pid_corrections();

  // 1.
  myFish->computeMidline(sim.time, sim.dt);
  myFish->computeSurface();

  // 2. & 3.
  integrateMidline();

  //CAREFUL: this func assumes everything is already centered around CM to start with, which is true (see steps 2. & 3. ...) for rX, rY: they are zero at CM, negative before and + after

  // 4. & 5.
  std::vector<VolumeSegment_OBB> vSegments = prepare_vSegments();

  // 6. & 7.
  const intersect_t segmPerBlock = prepare_segPerBlock(vSegments);
  assert(segmPerBlock.size() == obstacleBlocks.size());

  // 8.
  writeSDFOnBlocks(segmPerBlock);
}

void Fish::finalize()
{
  myFish->surfaceToComputationalFrame(_2Dangle, centerOfMass);
}

void Fish::update()
{
  // update position and angles
  Obstacle::update();
  angvel_integral[0] += sim.dt * angVel[0];
  angvel_integral[1] += sim.dt * angVel[1];
  angvel_integral[2] += sim.dt * angVel[2];
  #ifdef RL_LAYER
  auto P = 2*(myFish->timeshift-myFish->time0/myFish->l_Tp) +myFish->phaseShift;
  sr.phaseShift = fmod(P,2)<0 ? 2+fmod(P,2) : fmod(P,2);
  #endif
}

void Fish::save(std::string filename)
{
    //assert(std::abs(t-sim_time)<std::numeric_limits<Real>::epsilon());
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename + ".txt");

    savestream<<sim.time<<"\t"<<sim.dt<<std::endl;
    savestream<<position[0]<<"\t"<<position[1]<<"\t"<<position[2]<<std::endl;
    savestream<<quaternion[0]<<"\t"<<quaternion[1]<<"\t"<<quaternion[2]<<"\t"<<quaternion[3]<<std::endl;
    savestream<<transVel[0]<<"\t"<<transVel[1]<<"\t"<<transVel[2]<<std::endl;
    savestream<<angVel[0]<<"\t"<<angVel[1]<<"\t"<<angVel[2]<<std::endl;
    savestream<<theta_internal<<"\t"<<angvel_internal<<std::endl; // <<"\t"<<adjTh
    savestream<<_2Dangle;
    savestream.close();
}

void Fish::restart(std::string filename)
{
  std::ifstream restartstream;
  restartstream.open(filename+".txt");
  if(!restartstream.good()){
    printf("Could not restart from file\n");
    return;
  }
  Real restart_time, restart_dt;
  restartstream >> restart_time >> restart_dt;
  restartstream >> position[0] >> position[1] >> position[2];
  restartstream >> quaternion[0] >> quaternion[1] >> quaternion[2] >> quaternion[3];
  restartstream >> transVel[0] >> transVel[1] >> transVel[2];
  restartstream >> angVel[0] >> angVel[1] >> angVel[2];
  restartstream >> theta_internal >> angvel_internal;// >> adjTh;
  restartstream >> _2Dangle;
  restartstream.close();

  std::cout<<"RESTARTED FISH: "<<std::endl;
  std::cout<<"TIME, DT: "<<restart_time<<" "<<restart_dt<<std::endl;
  std::cout<<"POS: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<std::endl;
  std::cout<<"ANGLE: "<<quaternion[0]<<" "<<quaternion[1]
           <<" "<<quaternion[2]<<" "<<quaternion[3]<<std::endl;
  std::cout<<"TVEL: "<<transVel[0]<<" "<<transVel[1]<<" "<<transVel[2]<<std::endl;
  std::cout<<"AVEL: "<<angVel[0]<<" "<<angVel[1]<<" "<<angVel[2]<<std::endl;
  std::cout<<"INTERN: "<<theta_internal<<" "<<angvel_internal<<std::endl;
  std::cout<<"2D angle: \t"<<_2Dangle<<std::endl;
}

#ifdef RL_LAYER

void Fish::getSkinsAndPOV(Real& x, Real& y, Real& th,
  Real*& pXL, Real*& pYL, Real*& pXU, Real*& pYU, int& Npts)
{
  if( std::fabs(quaternion[1])>2e-16 || std::fabs(quaternion[2])>2e-16 ) {
    printf("the fish skin works only if the fish angular velocity is limited to the z axis. Aborting"); fflush(NULL);
    abort();
  }
  x  = position[0];
  y  = position[1];
  th  = _2Dangle;
  pXL = myFish->lowerSkin->xSurf;
  pYL = myFish->lowerSkin->ySurf;
  pXU = myFish->upperSkin->xSurf;
  pYU = myFish->upperSkin->ySurf;
  Npts = myFish->lowerSkin->Npoints;
}

void Fish::interpolateOnSkin(bool _dumpWake)
{
  if( std::fabs(quaternion[1])>2e-16 || std::fabs(quaternion[2])>2e-16 ) {
    printf("the fish skin works only if the fish angular velocity is limited to the z axis. Aborting"); fflush(NULL);
    abort();
  }
  sr.updateStepId(stepID+obstacleID);
  myFish->computeSkinNormals(_2Dangle, CoM_interpolated);

  sr.nearestGridPoints(obstacleBlocks, vInfo, myFish->upperSkin->Npoints-1,
                myFish->upperSkin->midX,      myFish->upperSkin->midY,
                myFish->lowerSkin->midX,      myFish->lowerSkin->midY,
                myFish->upperSkin->normXSurf, myFish->upperSkin->normYSurf,
                myFish->lowerSkin->normXSurf, myFish->lowerSkin->normYSurf,
                position[2], vInfo[0].h_gridpoint, grid->getCartComm());

  //  if(sim.rank==0) sr.print(obstacleID, stepID, time);

  //if(_dumpWake && _uInf not_eq nullptr) dumpWake(stepID, time, _uInf);
}

#endif // RL_LAYER

CubismUP_3D_NAMESPACE_END
