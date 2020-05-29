//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Schedulers_h
#define CubismUP_3D_Schedulers_h

#include "Interpolation1D.h"

#include <cmath>
#include <array>
#include <iostream>  // Schedulers.h not included from many files.
#include <fstream>   // Schedulers.h not included from many files.

CubismUP_3D_NAMESPACE_BEGIN

#define __BSPLINE

namespace Schedulers
{
template<int Npoints>
struct ParameterScheduler
{
  std::array<double, Npoints> parameters_t0; // parameters at t0
  std::array<double, Npoints> parameters_t1; // parameters at t1
  std::array<double, Npoints> dparameters_t0; // derivative at t0
  double t0, t1; // t0 and t1

  void save(std::string filename)
  {
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename+".txt");

    savestream << t0 << "\t" << t1 << std::endl;
    for(int i=0;i<Npoints;++i)
      savestream << parameters_t0[i]  << "\t"
                 << parameters_t1[i]  << "\t"
                 << dparameters_t0[i] << std::endl;
    savestream.close();
  }

  void restart(std::string filename)
  {
    std::ifstream restartstream;
    restartstream.open(filename+".txt");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(!rank) std::cout << filename << " ";
    restartstream >> t0 >> t1;
    for(int i=0;i<Npoints;++i) {
      restartstream >> parameters_t0[i] >> parameters_t1[i] >> dparameters_t0[i];
      if(!rank)
      std::cout<<parameters_t0[i]<<" "<<parameters_t1[i]<<" "<<dparameters_t0[i];
    }
    if(!rank) std::cout << std::endl;
    restartstream.close();
  }

  ParameterScheduler()
  {
    t0=-1; t1=0;
    parameters_t0 = std::array<double, Npoints>();
    parameters_t1 = std::array<double, Npoints>();
    dparameters_t0 = std::array<double, Npoints>();
  }

  void transition(const double t, const double tstart, const double tend,
      const std::array<double, Npoints> parameters_tend,
      const bool UseCurrentDerivative = false)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    //if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // we transition from whatever state we are in to a new state
    // the start point is where we are now: lets find out
    std::array<double, Npoints> parameters;
    std::array<double, Npoints> dparameters;
    gimmeValues(tstart,parameters,dparameters);

    /*
            if (Npoints == 7)
                printf("[t0 t1] were [%f %f], will be [%f %f], parameters %f dpdt %f param end %f\n",
                    t0,t1,tstart,tend,parameters[3],dparameters[3], parameters_tend[3]);
     */

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters;
    parameters_t1 = parameters_tend;
    dparameters_t0 = UseCurrentDerivative ? dparameters : std::array<double, Npoints>();
  }

  void transition(const double t, const double tstart, const double tend,
      const std::array<double, Npoints> parameters_tstart,
      const std::array<double, Npoints> parameters_tend)
  {
    if(t<tstart or t>tend) return; // this transition is out of scope
    if(tstart<t0) return; // this transition is not relevant: we are doing a next one already

    // fill my members
    t0 = tstart;
    t1 = tend;
    parameters_t0 = parameters_tstart;
    parameters_t1 = parameters_tend;
  }

  void gimmeValues(const double t, std::array<double, Npoints>& parameters, std::array<double, Npoints>& dparameters)
  {
    // look at the different cases
    if(t<t0 or t0<0) { // no transition, we are in state 0
      parameters = parameters_t0;
      dparameters = std::array<double, Npoints>();
    } else if(t>t1) { // no transition, we are in state 1
      parameters = parameters_t1;
      dparameters = std::array<double, Npoints>();
    } else { // we are within transition: interpolate
      for(int i=0;i<Npoints;++i)
        Interpolation1D::cubicInterpolation(t0,t1,t,parameters_t0[i],parameters_t1[i],dparameters_t0[i],0.0,parameters[i],dparameters[i]);
    }
  }

  void gimmeValues(const double t, std::array<double, Npoints>& parameters)
  {
    std::array<double, Npoints> dparameters_whocares; // no derivative info
    return gimmeValues(t,parameters,dparameters_whocares);
  }
};

struct ParameterSchedulerScalar : ParameterScheduler<1>
{
  void transition(const double t, const double tstart, const double tend, const double parameter_tend, const bool UseCurrentDerivative = false)
  {
    const std::array<double, 1> myParameter = {parameter_tend};
    return ParameterScheduler<1>::transition(t,tstart,tend,myParameter,UseCurrentDerivative);
  }

  void gimmeValues(const double t, double & parameter, double & dparameter)
  {
    std::array<double, 1> myParameter, mydParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter, mydParameter);
    parameter = myParameter[0];
    dparameter = mydParameter[0];
  }

  void gimmeValues(const double t, double & parameter)
  {
    std::array<double, 1> myParameter;
    ParameterScheduler<1>::gimmeValues(t, myParameter);
    parameter = myParameter[0];
  }
};

template<int Npoints>
struct ParameterSchedulerVector : ParameterScheduler<Npoints>
{
  void gimmeValues(const double t, const std::array<double, Npoints> & positions, const int Nfine,
      const Real * const positions_fine, Real * const parameters_fine, Real * const dparameters_fine)
  {
    // we interpolate in space the start and end point
    Real* parameters_t0_fine  = new Real[Nfine];
    Real* parameters_t1_fine  = new Real[Nfine];
    Real* dparameters_t0_fine = new Real[Nfine];

    Interpolation1D::naturalCubicSpline(positions.data(), this->parameters_t0.data(), Npoints, positions_fine, parameters_t0_fine,  Nfine);
    Interpolation1D::naturalCubicSpline(positions.data(), this->parameters_t1.data(), Npoints, positions_fine, parameters_t1_fine,  Nfine);
    Interpolation1D::naturalCubicSpline(positions.data(), this->dparameters_t0.data(),Npoints, positions_fine, dparameters_t0_fine, Nfine);

    // look at the different cases
    if(t<this->t0 or this->t0<0) {
      // no transition, we are in state 0
      for(int i=0;i<Nfine;++i) {
        parameters_fine[i] = parameters_t0_fine[i];
        dparameters_fine[i] = 0.0;
      }
    } else if(t>this->t1) {
      // no transition, we are in state 1
      for(int i=0;i<Nfine;++i) {
        parameters_fine[i] = parameters_t1_fine[i];
        dparameters_fine[i] = 0.0;
      }
    } else {
      // we are within transition: interpolate in time for each point of the fine discretization
      for(int i=0;i<Nfine;++i)
        Interpolation1D::cubicInterpolation(this->t0,this->t1,t,parameters_t0_fine[i],parameters_t1_fine[i],dparameters_t0_fine[i],
            0.0,           parameters_fine[i],   dparameters_fine[i]);
    }
    delete [] parameters_t0_fine;
    delete [] parameters_t1_fine;
    delete [] dparameters_t0_fine;
  }

  void gimmeValues(const double t, std::array<double, Npoints>& parameters)
  {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters);
  }

  void gimmeValues(const double t, std::array<double, Npoints> & parameters, std::array<double, Npoints> & dparameters)
  {
    ParameterScheduler<Npoints>::gimmeValues(t, parameters, dparameters);
  }
};

template<int Npoints>
struct ParameterSchedulerLearnWave : ParameterScheduler<Npoints>
{
  template<typename T>
  void gimmeValues(const double t, const double Twave, const double Length,
    const std::array<double, Npoints> & positions, const int Nfine,
    const T* const positions_fine, T* const parameters_fine, Real* const dparameters_fine)
  {
    const double _1oL = 1./Length;
    const double _1oT = 1./Twave;
    // the fish goes through (as function of t and s) a wave function that describes the curvature
    for(int i=0;i<Nfine;++i) {
      const double c = positions_fine[i]*_1oL - (t - this->t0)*_1oT; //traveling wave coord
      bool bCheck = true;

      if (c < positions[0]) { // Are you before latest wave node?
        Interpolation1D::cubicInterpolation(
          c, positions[0], c,
          this->parameters_t0[0], this->parameters_t0[0],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      }
      else if (c > positions[Npoints-1]) {// Are you after oldest wave node?
        Interpolation1D::cubicInterpolation(
          positions[Npoints-1], c, c,
          this->parameters_t0[Npoints-1], this->parameters_t0[Npoints-1],
          parameters_fine[i], dparameters_fine[i]);
        bCheck = false;
      } else {
        for (int j=1; j<Npoints; ++j) { // Check at which point of the travelling wave we are
          if (( c >= positions[j-1] ) && ( c <= positions[j] )) {
            Interpolation1D::cubicInterpolation(
              positions[j-1], positions[j], c,
              this->parameters_t0[j-1], this->parameters_t0[j],
              parameters_fine[i], dparameters_fine[i]);
            dparameters_fine[i] = -dparameters_fine[i]*_1oT; // df/dc * dc/dt
            bCheck = false;
          }
        }
      }
      if (bCheck) { std::cout << "Ciaone2!" << std::endl; abort(); }
    }
  }

  void Turn(const double b, const double t_turn) // each decision adds a node at the beginning of the wave (left, right, straight) and pops last node
  {
    this->t0 = t_turn;
    for(int i=Npoints-1; i>1; --i)
        this->parameters_t0[i] = this->parameters_t0[i-2];
    this->parameters_t0[1] = b;
    this->parameters_t0[0] = 0;
  }
};
}

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Schedulers_h
