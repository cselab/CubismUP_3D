//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#ifndef CubismUP_3D_Fish_h
#define CubismUP_3D_Fish_h

#include "Obstacle.h"

CubismUP_3D_NAMESPACE_BEGIN

class FishMidlineData;
struct VolumeSegment_OBB;

class Fish: public Obstacle
{
 protected:
  FishMidlineData * myFish = nullptr;
  //phaseShift=0, phase=0,
  double Tperiod=0;
  double volume_internal=0, J_internal=0;
  double CoM_internal[2]={0,0}, vCoM_internal[2]={0,0};
  double theta_internal=0, angvel_internal=0, angvel_internal_prev=0;
  double angvel_integral[3] = {0,0,0};
  //double adjTh=0, adjDy=0;
  bool bCorrectTrajectory=false, bCorrectPosition=false;

  void integrateMidline();
  //void apply_pid_corrections();

  // first how to create blocks of segments:
  typedef std::vector<VolumeSegment_OBB> vecsegm_t;
  vecsegm_t prepare_vSegments();
  // second how to intersect those blocks of segments with grid blocks:
  // (override to create special obstacle blocks for local force balances)
  typedef std::vector<std::vector<VolumeSegment_OBB*>> intersect_t;
  virtual intersect_t prepare_segPerBlock(vecsegm_t& vSeg);
  // third how to interpolate on the grid given the intersections:
  virtual void writeSDFOnBlocks(const intersect_t& segPerBlock);

 public:
  Fish(SimulationData&s, cubism::ArgumentParser&p);
  ~Fish() override;
  void save(std::string filename = std::string()) override;
  void restart(std::string filename = std::string()) override;

  virtual void update() override;

  virtual void create() override;
  virtual void finalize() override;

  #ifdef RL_LAYER
    void getSkinsAndPOV(Real& x, Real& y, Real& th, Real*& pXL, Real*& pYL,
      Real*& pXU, Real*& pYU, int& Npts) override;

    void interpolateOnSkin(const double time, const int stepID, bool dumpWake=false) override;
  #endif
  //  void computeVelocities(const Real Uinf[3]) override
  //  {
  //    computeVelocities_forced(Uinf);
  //  }
  // void setTranslationVelocity(double UT[3]) override  { }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Fish_h
