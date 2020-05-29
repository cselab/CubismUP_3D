//
//  CubismUP_3D
//
//  Written by Guido Novati ( novatig@ethz.ch ).
//  Copyright (c) 2017 ETHZ. All rights reserved.
//

#ifndef CubismUP_3D_TaskLayer_h
#define CubismUP_3D_TaskLayer_h

//#define __DumpWakeStefan 9

#ifdef __RL_MPI_CLIENT
#include <random>
#include "Communicator.h"
#endif

#include "StateRewardData.h"
#include "../obstacles/ObstacleVector.h"

#include <Cubism/ArgumentParser.h>

CubismUP_3D_NAMESPACE_BEGIN

struct TaskLayer
{
  #ifdef __RL_MPI_CLIENT
  Communicator* const communicator;
  #endif
  const MPI_Comm mpi_comm;
  int nActions=0, nStates=0, mpirank=0, mpisize=0, step_id = 0;
  bool finished = false;

  IF3D_ObstacleVector* obstacles = nullptr;
  std::vector<StateReward*> _D;

  #ifdef __RL_MPI_CLIENT
  TaskLayer(Communicator*const rl, const MPI_Comm mpi, ArgumentParser& parser):
    communicator(rl), mpi_comm(mpi)
  {
    MPI_Comm_rank(mpi, &mpirank);
    MPI_Comm_size(mpi, &mpisize);
    nStates = communicator->nStates;
    nActions = communicator->nActions;
    assert(nActions == 1 || nActions == 2);
    assert(nStates == (nActions==1) ? 20 + 4*NpLatLine : 25 + 4*NpLatLine);
  }
  #else
  TaskLayer(ArgumentParser& parser): mpi_comm(MPI_COMM_WORLD)
  {
    MPI_Comm_rank(mpi_comm, &mpirank);
    MPI_Comm_size(mpi_comm, &mpisize);
  }
  #endif

  void initializeObstacles(IF3D_ObstacleVector* const obst)
  {
    assert(obst not_eq nullptr);
    obstacles = obst;
    _D = obstacles->_getData();
    #ifdef __RL_MPI_CLIENT
      randomizeInitialPositions(); //needs comm
    #endif
  }

  double pos_x_frame = 0, pos_y_frame = 0, theta_frame = 0;
  double vel_x_frame = 0, vel_y_frame = 0, anvel_frame = 0;
  void setRefFrm_2LeadsFollower()
  {
    assert(_D.size()>=2);
    pos_x_frame =  .5*(_D[0]->Xrel +_D[1]->Xrel );
    pos_y_frame =  .5*(_D[0]->Yrel +_D[1]->Yrel );
    theta_frame =  .5*(_D[0]->thExp+_D[1]->thExp);
    vel_x_frame =  .5*(_D[0]->vxExp+_D[1]->vxExp);
    vel_y_frame =  .5*(_D[0]->vyExp+_D[1]->vyExp);
    anvel_frame =  .5*(_D[0]->avExp+_D[1]->avExp);
    //obstacles->getFieldOfView(_D[iAgent].lengthscale); //field of view
  }
  void setRefFrm_1LeadFollower()
  {
    assert(_D.size()>=1); //const double theta_frame =  _D[0]->thExp + 0.15;
    pos_x_frame =  _D[0]->Xrel;  vel_x_frame =  _D[0]->VxInst;
    pos_y_frame =  _D[0]->Yrel;  vel_y_frame =  _D[0]->VyInst;
    theta_frame =  _D[0]->Theta; anvel_frame =  _D[0]->AvInst;
  }
  void setRefFrm_DCylFollower()
  {
    assert(_D.size()>=1);
    pos_x_frame = _D[0]->Xrel; pos_y_frame = _D[0]->Yrel;
    theta_frame = vel_x_frame = vel_y_frame = anvel_frame = 0;
  }

  int step(const double time, const int iAgent, const int iLabel)
  {
    #ifdef __RL_MPI_CLIENT
      bool bDoOver = _D[iAgent]->checkTerm(pos_x_frame, pos_y_frame, theta_frame,
                                           vel_x_frame, vel_y_frame, anvel_frame);
      if(_D[iAgent]->t_next_comm > time && not bDoOver) return 0;

      fflush(0);
      setRefFrm();
      //update sensors
      //obstacles->getFieldOfView(_D[iAgent]->lengthscale); //field of view
      obstacles->interpolateOnSkin(time, step_id++, iAgent);
      _D[iAgent]->finalize(pos_x_frame, pos_y_frame, theta_frame,
                           vel_x_frame, vel_y_frame, anvel_frame);

      const vector<double> state = _D[iAgent]->fillState(time,nStates,nActions);
      assert(state.size() == nStates);
      const double reward = getReward(iAgent);
      communicator->sendState(iLabel, _D[iAgent]->info, state, reward);
      if (_D[iAgent]->info==2) {
        finished = true;
        return 1;
      }
      _D[iAgent]->info = 0;
      const vector<double> action = communicator->recvAction();
      obstacles->execute(iAgent, time, action);
    #endif

    return 0;
  }

  int operator()(const int stepnum, const double time)
  {
    if(!stepnum) return 0;
    int iLabel = 0;
    for(size_t i=0; i<_D.size(); i++) {
      if(_D[i]->bInteractive) step(time, i, iLabel++);
    }
    return finished;
  }

  double getReward(const int iAgent)
  {
    return _D[iAgent]->info==2 ? -10 : _D[iAgent]->EffPDefBnd;
  }

  #ifdef __RL_MPI_CLIENT
  void randomizeInitialPositions()
  {
    const std::vector<IF3D_ObstacleOperator*> o=obstacles->getObstacleVector();
    for(size_t i=0; i<_D.size(); i++)
    {
      if(not _D[i]->randomStart) continue;

      double init[4];
      if (!mpirank) {
        std::uniform_real_distribution<double> dis(-1.,1.);
        init[0] = dis(communicator->gen);
        init[1] = dis(communicator->gen);
        init[2] = dis(communicator->gen);
        init[3] = dis(communicator->gen);

        for(int i=1; i<mpisize; i++)
          MPI_Send(init,4,MPI_DOUBLE,i,9,mpi_comm);
      } else
          MPI_Recv(init,4,MPI_DOUBLE,0,9,mpi_comm,MPI_STATUS_IGNORE);

      printf("Rank %d (out of %d) using seeds %g %g %g %g\n",
        mpirank, mpisize, init[0], init[1], init[2], init[3]);
      fflush(0);
      sendInitC(i, init, o);
      _D[i]->updateInstant(
        o[i]->position[0], o[i]->absPos[0], o[i]->position[1], o[i]->absPos[1],
        o[i]->_2Dangle, o[i]->transVel[0], o[i]->transVel[1], o[i]->angVel[2]);
    }
  }
  #endif

  void sendInitC_LeaderFollower(const int iAgent, double init[4],
    const std::vector<IF3D_ObstacleOperator*>& o)
  {
    #ifdef __ExploreHalfWake //explore only one half of the domain:
      init[1] = std::fabs(init[1]);
    #endif
    o[iAgent]->position[0] += .5 * o[iAgent]->length * init[0];
    const double dX = o[iAgent]->position[0] - o[0]->position[0];
    //now adding a shift so that i do not over explore dy = 0;
    const double shiftDy = init[1]>0 ? (0.25*dX/2.5) : -(0.25*dX/2.5);
    o[iAgent]->position[1] += .25 * o[iAgent]->length * init[1] + shiftDy;
    o[iAgent]->absPos[0] = o[iAgent]->position[0];
    o[iAgent]->absPos[1] = o[iAgent]->position[1];
    o[iAgent]->_2Dangle = .1* M_PI *init[2];
    _D[iAgent]->thExp = o[iAgent]->_2Dangle;

    o[iAgent]->quaternion[0] = std::cos(0.5*o[iAgent]->_2Dangle); o[iAgent]->quaternion[1] = 0;
    o[iAgent]->quaternion[2] = 0; o[iAgent]->quaternion[3] = std::sin(0.5*o[iAgent]->_2Dangle);
    //if(nActions==2)
    o[iAgent]->phaseShift = init[3];
  }
  void sendInitC_DcylFollower(const int iAgent, double init[4],
    const std::vector<IF3D_ObstacleOperator*>& o)
  {
    o[iAgent]->position[0] += .50 * o[iAgent]->length * init[0];
    o[iAgent]->position[1] += .25 * o[iAgent]->length * init[1];
    o[iAgent]->absPos[0]  = o[iAgent]->position[0];
    o[iAgent]->absPos[1]  = o[iAgent]->position[1];
    o[iAgent]->_2Dangle = .1* M_PI *init[2];
    _D[iAgent]->thExp = o[iAgent]->_2Dangle;
    o[iAgent]->quaternion[0] = std::cos(0.5*o[iAgent]->_2Dangle);
    o[iAgent]->quaternion[1] = 0;
    o[iAgent]->quaternion[2] = 0;
    o[iAgent]->quaternion[3] = std::sin(0.5*o[iAgent]->_2Dangle);
    //if(nActions==2)
    o[iAgent]->phaseShift = init[3];
  }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_TaskLayer_h
