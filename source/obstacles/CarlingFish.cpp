//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CarlingFish.h"
#include "FishLibrary.h"
#include "FishShapes.h"

#include <Cubism/ArgumentParser.h>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

class CarlingFishMidlineData : public FishMidlineData
{
 public:
  bool quadraticAmplitude = false;
 protected:
  const double carlingAmp;
  static constexpr double carlingInv = 0.03125;

  const double quadraticFactor; // Should be set to 0.1, which gives peak-to-peak amp of 0.2L (this is physically observed in most fish species)

  inline Real rampFactorSine(const Real t, const Real T) const
  {
    //return (t<T ? ( 1 - std::cos(M_PI*t/T) )/2 : 1.0);
    return (t<T ? std::sin(0.5*M_PI*t/T) : 1.0);
  }

  inline Real rampFactorVelSine(const Real t, const Real T) const
  {
    //return (t<T ? 0.5*M_PI/T * std::sin(M_PI*t/T) : 0.0);
    return (t<T ? 0.5*M_PI/T * std::cos(0.5*M_PI*t/T) : 0.0);
  }

  inline Real getQuadAmp(const Real s) const
  {
    // Maertens et al. JFM 2017:
    return quadraticFactor*(length -.825*(s-length) +1.625*(s*s/length-length));
    //return s*s*quadraticFactor/length;
  }
  inline Real getLinAmp(const Real s) const
  {
    return carlingAmp * (s + carlingInv*length);
  }

  inline Real getArg(const Real s,const Real t) const
  {
    return 2.0*M_PI*(s/(waveLength*length) - t/Tperiod + phaseShift);
  }

  // This needed only during burstCoast
  std::pair<double, double> cubicHermite(const double f1, const double f2, const double x)
  {
    const double a =  2*(f1-f2);
    const double b = -3*(f1-f2);
    const double retVal = a*x*x*x + b*x*x + f1;
    const double deriv = 3*a*x*x + 2*b*x;
    return std::make_pair(retVal, deriv);
  }

 public:
  // L=length, T=period, phi=phase shift, _h=grid size, A=amplitude modulation
  CarlingFishMidlineData(double L, double T, double phi, double _h, double A) :
  FishMidlineData(L,T,phi,_h,A),carlingAmp(.1212121212*A),quadraticFactor(.1*A)
  {
    // FinSize has now been updated with value read from text file. Recompute heights to over-write with updated values
    //printf("Overwriting default tail-fin size for Plain Carling:\n");
    //_computeWidthsHeights();
  }

  virtual void computeMidline(const double t, const double dt) override;

  template<bool bQuadratic>
  void _computeMidlinePosVel(const Real t)
  {
    const Real rampFac = rampFactorSine(t, Tperiod), dArg = -2*M_PI/Tperiod;
    const Real rampFacVel = rampFactorVelSine(t, Tperiod);
    {
      const Real arg = getArg(rS[0], t);
      const Real cosa = std::cos(arg), sina = std::sin(arg);
      const Real amp = bQuadratic? getQuadAmp(rS[0]) : getLinAmp(rS[0]);
      const Real Y = sina * amp, VY = cosa * dArg * amp;
      rX[0] = 0.0; vX[0] = 0.0; //rX[0] is constant
      rY[0] = rampFac*Y;
      vY[0] = rampFac*VY + rampFacVel*Y;
    }
    for(int i=1; i<Nm; ++i)
    {
      const Real arg = getArg(rS[i], t);
      const Real cosa = std::cos(arg), sina = std::sin(arg);
      const Real amp = bQuadratic? getQuadAmp(rS[i]) : getLinAmp(rS[i]);
      const Real Y = sina * amp, VY = cosa * dArg * amp;
      rY[i] = rampFac*Y;
      vY[i] = rampFac*VY + rampFacVel*Y;
      const Real dy = rY[i]-rY[i-1], ds = rS[i]-rS[i-1], dVy = vY[i]-vY[i-1];
      const Real dx = std::sqrt(ds*ds-dy*dy);
      assert(dx>0);
      rX[i] = rX[i-1] + dx;
      vX[i] = vX[i-1] - dy/dx *dVy; // use ds^2 = dx^2+dy^2 -> ddx = -dy/dx*ddy
    }
  }
};

#include "extra/CarlingFish_extra.h"

void CarlingFishMidlineData::computeMidline(const double t,const double dt)
{
  if(quadraticAmplitude) _computeMidlinePosVel<true >(t);
  else                   _computeMidlinePosVel<false>(t);

  _computeMidlineNormals();
  #if 0
    #warning USED MPI COMM WORLD
    // we dump the profile
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank!=0) return;
    FILE * f = fopen("fish_profile","w");
    for(int i=0;i<Nm;++i) fprintf(f,"%d %g %g %g %g %g %g %g %g %g\n",
     i,rS[i],rX[i],rY[i],norX[i],norY[i],vX[i],vY[i], vNorX[i],vNorY[i]);
    fclose(f); printf("Dumped midline\n");
  #endif
}

CarlingFish::CarlingFish(SimulationData&s, ArgumentParser&p) : Fish(s, p)
{
  // _ampFac=0.0 for towed fish :
  const double ampFac = p("-amplitudeFactor").asDouble(1.0);
  const bool bQuadratic = p("-bQuadratic").asBool(false);
  const bool bBurst = p("-BurstCoast").asBool(false);
  const bool bHinge = p("-HingedFin").asBool(false);
  if(bBurst && bHinge) {
    printf("Pick either hinge or burst and coast!\n"); fflush(0);
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if(bBurst || bHinge) printf("WARNING: UNTESTED!!!\n");

  CarlingFishMidlineData* localFish = nullptr; //could be class var if needed
  if(bBurst) localFish = readBurstCoastParams(p);
  else
  if(bHinge) localFish = readHingeParams(p);
  else
  localFish = new CarlingFishMidlineData(length, Tperiod, phaseShift,
    sim.maxH(), ampFac);

  // generic copy for base class:
  assert( myFish == nullptr );
  myFish = (FishMidlineData*) localFish;

  localFish->quadraticAmplitude = bQuadratic;
  std::string heightName = p("-heightProfile").asString("baseline");
  std::string  widthName = p( "-widthProfile").asString("baseline");
  MidlineShapes::computeWidthsHeights(heightName, widthName, length,
    myFish->rS, myFish->height, myFish->width, myFish->Nm, sim.rank);

  if(!sim.rank)
    printf("CarlingFish: N:%d, L:%f, T:%f, phi:%f, amplitude:%f\n",
        myFish->Nm, length, Tperiod, phaseShift, ampFac);
}

CubismUP_3D_NAMESPACE_END
