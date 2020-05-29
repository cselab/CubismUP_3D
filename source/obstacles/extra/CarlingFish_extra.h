//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_CarlingFish_extra_h
#define CubismUP_3D_CarlingFish_extra_h

class CarlingFishMidlineData_BurstCoast : public CarlingFishMidlineData
{
 protected:
  const Real tStart, t0, t1, t2, t3, lowestAmp;

  Real midlineBC(const Real s, const Real t, const Real f) const
  {
    return f * getLinAmp(s) * std::sin(getArg(s,t));
  }

  Real midlineVelBC(const Real s, const Real t, const Real f, const Real df) const
  {
    const Real arg = getArg(s,t), dArg = (2*M_PI/Tperiod)*std::cos(arg);
    return getLinAmp(s)*(df*std::sin(arg) - f*dArg);
  }

 public:
  void computeMidline(const double t, const double dt) override;

  CarlingFishMidlineData_BurstCoast(double _tStart, double T0, double T1,
    double T2, double T3, double lowAmp, double L, double T, double phi,
    double _h, double A) : CarlingFishMidlineData(L,T,phi,_h,A),
    tStart(_tStart), t0(T0), t1(T1), t2(T2), t3(T3), lowestAmp(lowAmp)
  {
    printf("CarlingFishMidlineData_BurstCoast NOT SUPPORTED\n");
  }
};

class CarlingFishMidlineData_Hinged : public CarlingFishMidlineData
{
 protected:
  const Real sHinge, ThingeTheta, AhingeTheta, hingePhi;

  //Real kSpring=0.0;
  //const Real kMaxSpring=100.0; // Expect torque values on the order of 1e-5 at steady, and 1e-3 at startup
  //Real thetaOld = 0.0, avgTorque = 0.0, runningTorque = 0.0, timeNminus = 0.0;
  //int prevTransition = 0;

  const Real sLeft  = sHinge - 0.02*length;
  const Real sRight = sHinge + 0.02*length;

  Real getJointParabola(const Real s, const Real aParabola,
    const Real bParabola, const Real cParabola) const
  {
    return aParabola*s*s + bParabola*s + cParabola;
  }

  Real midline(const Real s, const Real t) const;

  Real midlineVel(const Real s, const Real t) const;

 public:
  CarlingFishMidlineData_Hinged(double _sHinge,double _Ahinge,double _phiHinge,
   double _Thinge, double L,double T,double phi,double _h,double A) :
   CarlingFishMidlineData(L,T,phi,_h,A), sHinge(_sHinge), ThingeTheta(_Thinge),
    AhingeTheta(M_PI*_Ahinge/180.0), hingePhi(_phiHinge/360.0)
  {
    printf("CarlingFishMidlineData_Hinged NOT SUPPORTED\n");
  }
};

void CarlingFishMidlineData_BurstCoast::computeMidline(const double t, const double dt)
{
  const Real rampFac = rampFactorSine(t, Tperiod);
  const Real rampFacVel = rampFactorVelSine(t, Tperiod);

  Real f, df;
  const Real bct     = t0 + t1 + t2 + t3;
  assert(bct>0);
  const Real shift   = std::floor((t-tStart)/bct);
  const Real tcoast  = tStart  + shift*bct;
  const Real tfreeze = tcoast  + t0;
  const Real tburst  = tfreeze + t1;
  const Real tswim   = tburst  + t2;
  //const Real phase   = (time<tfreeze) ?  shift   *0.5 + phaseShift
  //           : (shift+1)*0.5 + phaseShift;

  if (t<tcoast) {
    f = 1.0;
    df = 0.0;
  } else if (t<tfreeze) {
    const Real d = (t-tcoast)/(tfreeze-tcoast);
    const std::pair<double, double> retVal = cubicHermite(1.0, lowestAmp, d);
    f = retVal.first;
    df = retVal.second/(tfreeze-tcoast);
    //f = 1 - 3*d*d + 2*d*d*d;
  } else if (t<tburst) {
    //f = 0.0;
    f = lowestAmp;
    df = 0.0;
  } else if (t<tswim) {
    const Real d = (t-tburst)/(tswim-tburst);
    const std::pair<double, double> retVal = cubicHermite(lowestAmp, 1.0, d);
    f = retVal.first;
    df = retVal.second/(tswim-tburst);
    //f = 3*d*d - 2*d*d*d;
    //df = 6*(d - d*d)/(tswim-tburst);
  } else {
    f = 1.0;
    df = 0.0;
  }

  rX[0] = 0.0;
  rY[0] = rampFac*midlineBC(rS[0], t, f);
  for(int i=1; i<Nm; ++i)
  {
    rY[i] = rampFac*midlineBC(rS[i], t, f);
    const Real dy = rY[i]-rY[i-1], ds = rS[i] - rS[i-1];
    rX[i] = rX[i-1] + std::sqrt(ds*ds-dy*dy);
  }

  vX[0] = 0.0; //rX[0] is constant
  vY[0] = rampFac*midlineVelBC(rS[0],t,f,df) + rampFacVel*midlineBC(rS[0],t,f);
  for(int i=1; i<Nm; ++i)
  {
    vY[i]=rampFac*midlineVelBC(rS[i],t,f,df) + rampFacVel*midlineBC(rS[i],t,f);
    const Real dy  = rY[i]-rY[i-1], dx  = rX[i]-rX[i-1], dVy = vY[i]-vY[i-1];
    assert(dx>0); //has to be, otherwise y(s) is multiple valued for a given s
    vX[i] = vX[i-1] - dy/dx * dVy; // use ds^2 = dx^2+dy^2 -> ddx = -dy/dx*ddy
  }
}

Real CarlingFishMidlineData_Hinged::midline(const Real s, const Real t) const
{
  double yCurrent = 0;//CarlingFishMidlineData::midline(s, t);

  if(s >= sLeft)
  {
    const Real argLeft = getArg(sLeft, t), ampLeft = getQuadAmp(sLeft);
    const double yLeft = ampLeft *std::sin(argLeft);
    const Real dAmp = 2*quadraticFactor*sLeft/length;
    const Real dArg = std::cos(argLeft) * 2*M_PI/(length*waveLength);
    const double yPrimeLeft = dAmp*std::sin(argLeft) + ampLeft*dArg;

    const double thetaArg = 2*M_PI*(t/ThingeTheta + hingePhi);
    const double currentTheta = AhingeTheta * std::sin(thetaArg);
    const double yPrimeRight = std::sin(currentTheta);

    const double aParab = (yPrimeLeft-yPrimeRight)/(2*(sLeft-sRight));
    const double bParab = yPrimeRight - 2*aParab*sRight;
    const double cParab = yLeft - aParab*sLeft*sLeft - bParab*sLeft;

    yCurrent = getJointParabola(s,aParab,bParab,cParab);

    if(s>=sRight)
    {
      const Real yRight = getJointParabola(sRight,aParab,bParab,cParab);
      yCurrent = yRight + yPrimeRight*(s-sRight);
    }
  }
  return yCurrent;
  /*if(s>sHinge){
    double yNot;
    if(quadraticAmplitude){
      yNot =  (sHinge*sHinge*quadraticFactor/L)*std::sin(2.0*M_PI*(sHinge/(waveLength*L) - t/T + phaseShift));
    }else{
      yNot =  fac *  (sHinge + inv*L)*std::sin(2.0*M_PI*(sHinge/(waveLength*L) - t/T + phaseShift));
    }
    const double currentTheta = AhingeTheta * std::sin(2.0*M_PI*(t/ThingeTheta + hingePhi));
    const double dydsNot = std::sin(currentTheta);
    yCurrent = yNot + dydsNot*(s-sHinge);
  }*/
}

Real CarlingFishMidlineData_Hinged::midlineVel(const Real s, const Real t) const
{
  double velCurrent = 0; //CarlingFishMidlineData::midlineVel(s, t);
  if(s>=sLeft)
  {
    const Real argLeft = getArg(sLeft, t), ampLeft = getQuadAmp(sLeft);
    const double yLeft = ampLeft *std::sin(argLeft);
    const Real dAmp = 2*quadraticFactor*sLeft/length;
    const Real ddArg = yLeft * 2*M_PI/(length*waveLength);

    const double yLeftDot  = (-2*M_PI/Tperiod) * ampLeft*std::cos(argLeft);
    const double yPrimeLeftDot = dAmp * yLeftDot + ddArg *(2*M_PI/Tperiod);

    const double thetaArg = 2*M_PI*(t/ThingeTheta + hingePhi);
    const double dThetaArg = 2*M_PI/ThingeTheta;
    const double currentTheta = AhingeTheta * std::sin(thetaArg);
    const double currentThetaDot = AhingeTheta * dThetaArg*std::cos(thetaArg);

    const double yPrimeRightDot = std::cos(currentTheta)*currentThetaDot;

    const double aDot = (yPrimeLeftDot - yPrimeRightDot)/(2*(sLeft-sRight));
    const double bDot = yPrimeRightDot - 2*sRight*aDot;
    const double cDot = yLeftDot - sLeft*sLeft*aDot - sLeft*bDot;
    velCurrent = getJointParabola(s,aDot,bDot,cDot);

    if(s>=sRight)
    {
      const Real yRightDot = getJointParabola(s,aDot,bDot,cDot);
      velCurrent = yRightDot + yPrimeRightDot*(s-sRight);
    }
  }
  return velCurrent;
  /*
  if(s>sHinge) {
    //const double yNot =  4./33 *  (sHinge + 0.03125*L)*std::sin(2.0*M_PI*(sHinge/L - t/T + phaseShift));
    double velNot;
    double velNot;
    if(quadraticAmplitude){
      velNot =  -2.0*M_PI/T * (sHinge*sHinge*quadraticFactor/L)
        *std::cos(2.0*M_PI*(sHinge/(L*waveLength) - t/T + phaseShift));
    }else{
      velNot =  -2.0*M_PI/T * fac *  (sHinge + inv*L)*
      std::cos(2.0*M_PI*(sHinge/(L*waveLength) - t/T + phaseShift));
    }
    const double currentTheta = AhingeTheta * std::sin(2.0*M_PI*(t/ThingeTheta + hingePhi));
    const double currentThetaDot = AhingeTheta * 2.0*M_PI/ThingeTheta * std::cos(2.0*M_PI*(t/ThingeTheta + hingePhi));
    const double dydsNotDT = std::cos(currentTheta)*currentThetaDot;
    velCurrent = velNot + dydsNotDT*(s-sHinge);
  }
  */
}

CarlingFishMidlineData* CarlingFish::readHingeParams(ArgumentParser&p)
{
  Real sHinge, aHinge, phiHinge, waveLength = 1, tHinge = Tperiod;
  sHinge = length*p("-sHinge").asDouble();
  const double ampFac = p("-amplitudeFactor").asDouble(1.0);
  //const bool equalHeight = length && parser("-equalHeight").asBool();

  const bool bOptimizeHinge = p("-OptimizeHingedFin").asBool(false);
  if(bOptimizeHinge)
  {
    std::ifstream reader("hingedParams.txt");
    if (reader.is_open()) {
      reader >> aHinge;
      reader >> phiHinge;
      reader >> waveLength;
      //reader >> finSize;
      printf("Read numbers = %f, %f, %f\n", aHinge, phiHinge, waveLength);
      if(reader.eof()){
        std::cout << "Insufficient number of parameters provided for hingedFin" << std::endl; fflush(NULL); abort();
      }
      reader.close();
    } else {
      std::cout << "Could not open the correct 'params'.txt file" << std::endl; fflush(NULL);
      abort();
    }
  }
  else
  {
    aHinge = p("-AhingeDeg").asDouble();
    phiHinge = p("-phiHingeDeg").asDouble();
  }

  return new CarlingFishMidlineData_Hinged(sHinge, aHinge, phiHinge, tHinge,
    length, Tperiod, phaseShift, sim.maxH(), ampFac);
}

CarlingFishMidlineData* CarlingFish::readBurstCoastParams(ArgumentParser&p)
{
  const Real tStart = p("-tStartBC").asDouble();
  const double ampFac = p("-amplitudeFactor").asDouble(1.0);
  Real t0, t1, t2, t3, lowestAmp;
  std::ifstream reader("burst_coast_carling_params.txt");
  if (reader.is_open()) {
    reader >> t0;
    reader >> t1;
    reader >> t2;
    reader >> t3;
    reader >> lowestAmp;
    if(reader.eof()){
      std::cout<<"Insufficient number of parameters provided for burstCoast"<<std::endl;
      abort();
    }
    reader.close();
  } else {
    std::cout << "Could not open the correct 'params'.txt file" << std::endl;
    fflush(NULL);
    abort();
  }
  return new CarlingFishMidlineData_BurstCoast(tStart, t0, t1, t2, t3,
    lowestAmp, length, Tperiod, phaseShift, sim.maxH(), ampFac);
}

#if 0 // COMPUTE MIDLINES AND VELS WITH DOUBLE HINGE
void _computeMidlineCoordinates(const Real time)
{
  const Real rampFac = rampFactorSine(time, Tperiod);
  rX[0] = 0.0;
  rY[0] = rampFac*midline(rS[0], time, length, Tperiod, phaseShift);

  int hinge1Index = -1, hinge2Index = -1;

  for(int i=1;i<Nm;++i) {
    rY[i]=rampFac*midline(rS[i], time, length, Tperiod, phaseShift);
    const Real dy = rY[i]-rY[i-1];
    const Real ds = rS[i] - rS[i-1];
    Real dx = std::sqrt(ds*ds-dy*dy);

    // dx can be undef for s<0 and s>L points when wavelength>1. I dunno why we have these goddamn s<0 and s>L points
    if(not(dx>0) and not(waveLength==1.0)){
      dx = 0.001*length;
    }

    rX[i] = rX[i-1] + dx;

    if(rS[i]>=sHinge and hinge1Index<0) hinge1Index = i;
    if(rS[i]>=sHinge2 and hinge2Index<0) hinge2Index = i;
  }

  // Now do the second hinge section
  if(bDoubleHinge){

    //linearly decrease spring stiffness over 1 Tperiod, otherwise might get discontinuous theta2 at startup
    /*const bool kSpringTransition = std::floor(time-Tperiod) < 0;
      const double kCurrent = not(kSpringTransition) ? kSpring : (kMaxSpring + time*(kSpring-kMaxSpring)/Tperiod);*/

    const double dt = time - this->timeNminus;
    this->timeNminus = time;
    this->runningTorque += this->torqueZsecMarkers * dt;

    if(time> (prevTransition+1)*0.01*Tperiod ){
      this->tOld = time;
      this->thetaOld = this->dTheta2;
      this->avgTorque = this->runningTorque/(0.01*Tperiod);
      this->runningTorque = 0.0;
      prevTransition++;
    }

    //Rigid until time period
    //if(time<Tperiod)
    if(1){
      this->dTheta2 = 0.0;
    }else{
      const double kCurrent = kSpring;
      const double cSpring = kCurrent;
      //const double thetaOld = this->dTheta2;
      const double thetaOld = this->thetaOld;
      const double tOld = this->tOld;
      const double torque = this->avgTorque;
      const double a1 = (thetaOld - torque/kCurrent)*exp(tOld*kCurrent/cSpring);
      //this->tOld = time;
      this->dTheta2 = torque/kCurrent + a1*exp(-time*kCurrent/cSpring);
      printf("time = %f, dTheta2 = %f, kSpring=%f, torque=%f\n", time, this->dTheta2, kCurrent, torque);
    }

    const double hinge1Loc[2] = {rX[hinge1Index], rY[hinge1Index]};
    const double hinge2Loc[2] = {rX[hinge2Index], rY[hinge2Index]};

    // angle of arm1 wrt main fish - imposed
    // Don't use analytical thetaExpression, since rampFactor not accounted-for in there
    const double currentTheta = std::atan( (hinge2Loc[1] - hinge1Loc[1]) / (hinge2Loc[0] - hinge1Loc[0]));

    for(int i=hinge2Index; i<Nm; ++i){
      const double dx = rX[i] - hinge2Loc[0];
      const double dy = rY[i] - hinge2Loc[1];
      const double localLength = std::sqrt(dx*dx + dy*dy);

      // angle of arm2 wrt main fish - from spring
      const double thetaHinge2 = currentTheta + dTheta2;

      rX[i] = hinge2Loc[0] + localLength*std::cos(thetaHinge2);
      rY[i] = hinge2Loc[1] + localLength*std::sin(thetaHinge2);
    }
  }
}

void _computeMidlineVelocities(const Real time)
{

  const Real rampFac =    rampFactorSine(time, Tperiod);
  const Real rampFacVel = rampFactorVelSine(time, Tperiod);

  vX[0] = 0.0; //rX[0] is constant
  vY[0] = rampFac*midlineVel(rS[0],time,length,Tperiod, phaseShift) +
      rampFacVel*midline(rS[0],time,length,Tperiod, phaseShift);

  int indHinge2 = -1;
  for(int i=1;i<Nm;++i) {
    vY[i]=rampFac*midlineVel(rS[i],time,length,Tperiod, phaseShift) +
        rampFacVel*midline(rS[i],time,length,Tperiod, phaseShift);
    const Real dy  = rY[i]-rY[i-1];
    const Real dx  = rX[i]-rX[i-1];
    const Real dVy = vY[i]-vY[i-1];

    vX[i] = vX[i-1] - dy/dx * dVy; // use ds^2 = dx^2 + dy^2 --> ddx = -dy/dx*ddy
    if(waveLength==1.0){
      assert(dx>0); // has to be, otherwise y(s) is multiple valued for a given s
    }else{ // dx can be undef for s<0 and s>L points when wavelength>1. I dunno why we have these goddamn s<0 and s>L points
      if(not(dx>0))  vX[i] = 0.0;
    }

    if(indHinge2<0 and rS[i]>=sHinge2) indHinge2 = i;
  }

  if(bDoubleHinge){

    double dtHinge = time-oldTime;

    if(firstStep){
      for(int i=indHinge2; i<Nm; ++i){
        rXold[i] = rX[i];
        rYold[i] = rY[i];
      }
      firstStep = false;
      dtHinge = 1.0; // To avoid divide by zero at first step
    }

    for(int i=indHinge2; i<Nm; ++i){
      vX[i] = (rX[i] - rXold[i]) / dtHinge;
      vY[i] = (rY[i] - rYold[i]) / dtHinge;

      rXold[i] = rX[i];
      rYold[i] = rY[i];
    }
  }
  oldTime = time;

  /*FILE *temp;
  temp = fopen("vels.txt","a");
  for (int i=0; i<Nm; ++i){
    fprintf(temp,"%f\t", rS[i]);
  }

  fprintf(temp,"\n");
  for (int i=0; i<Nm; ++i){
    fprintf(temp,"%f\t", rX[i]);
  }

  fprintf(temp,"\n");
  for (int i=0; i<Nm; ++i){
    fprintf(temp,"%f\t", rY[i]);
  }

  fprintf(temp,"\n");
  for (int i=0; i<Nm; ++i){
    fprintf(temp,"%f\t", vX[i]);
  }

  fprintf(temp,"\n");
  for (int i=0; i<Nm; ++i){
    fprintf(temp,"%f\t", vY[i]);
  }

  fprintf(temp,"\n");
  fclose(temp);*/

}
#endif // COMPUTE MIDLINES AND VELS WITH DOUBLE HINGE

#endif // CubismUP_3D_CarlingFish_extra_h
