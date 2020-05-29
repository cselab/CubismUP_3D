//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_ObstacleVector_h
#define CubismUP_3D_ObstacleVector_h

#include "Obstacle.h"

#include <memory>
#include <utility>

CubismUP_3D_NAMESPACE_BEGIN

class ObstacleVector : public Obstacle
{
 public:
    typedef std::vector<std::shared_ptr<Obstacle>> VectorType;

    ObstacleVector(SimulationData&s) : Obstacle(s) {}

    Obstacle * operator() (const size_t ind) const {
      return obstacles[ind].get();
    }
    int nObstacles() const {return obstacles.size();}
    void computeVelocities() override;
    void update() override;
    void restart(std::string filename = std::string()) override;
    void save(std::string filename = std::string()) override;

    void computeForces() override;

    void create() override;
    void finalize() override;
    void Accept(ObstacleVisitor * visitor) override;

    std::vector<std::array<int, 2>> collidingObstacles();

    void addObstacle(std::shared_ptr<Obstacle> obstacle)
    {
        obstacle->obstacleID = obstacles.size();
        obstacles.emplace_back(std::move(obstacle));
    }

    const VectorType& getObstacleVector() const
    {
        return obstacles;
    }

    std::vector<std::vector<ObstacleBlock*>*> getAllObstacleBlocks() const
    {
      const size_t Nobs = obstacles.size();
      std::vector<std::vector<ObstacleBlock*>*> ret(Nobs, nullptr);
      for(size_t i=0; i<Nobs; i++) ret[i]= obstacles[i]->getObstacleBlocksPtr();
      return ret;
    }
    Real getD() const override;

    std::array<Real,3> updateUinf() const
    {
      std::array<Real,3> nSum = {0, 0, 0};
      std::array<Real,3> uSum = {0, 0, 0};
      for(const auto & obstacle_ptr : obstacles) {
        const auto obstacle_fix = obstacle_ptr->bFixFrameOfRef;
        const auto obstacle_vel = obstacle_ptr->transVel;
        if (obstacle_fix[0]) { nSum[0]+=1; uSum[0] -= obstacle_vel[0]; }
        if (obstacle_fix[1]) { nSum[1]+=1; uSum[1] -= obstacle_vel[1]; }
        if (obstacle_fix[2]) { nSum[2]+=1; uSum[2] -= obstacle_vel[2]; }
      }
      if(nSum[0]>0) uSum[0] = uSum[0] / nSum[0];
      if(nSum[1]>0) uSum[1] = uSum[1] / nSum[1];
      if(nSum[2]>0) uSum[2] = uSum[2] / nSum[2];
      return uSum;
      //if(rank == 0) if(nSum[0] || nSum[1] || nSum[2])
      //  printf("New Uinf %g %g %g (from %d %d %d)\n",
      //  uInf[0],uInf[1],uInf[2],nSum[0],nSum[1],nSum[2]);
    }

    #ifdef RL_LAYER
      std::vector<StateReward*> _getData();

      void getFieldOfView(const double lengthscale);

      void execute(const int iAgent, const double time, const std::vector<double> action) override;

      void interpolateOnSkin(const double time, const int step, bool dumpWake=false) override;
    #endif

 protected:
    VectorType obstacles;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_ObstacleVector_h
