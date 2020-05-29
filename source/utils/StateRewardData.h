//
//  CubismUP_3D
//
//  Written by Guido Novati ( novatig@ethz.ch ).
//  Copyright (c) 2017 ETHZ. All rights reserved.
//

#ifndef CubismUP_3D_StateRewardData_h
#define CubismUP_3D_StateRewardData_h

//#include <cassert>
//#include <assert.h>

// utmost import to be defined before including cubism
static const int NpLatLine = 10;
//#define __ExploreHalfWake

#ifdef __RL_MPI_CLIENT //hardcoded BC for DCyl
#define checkTerm(...) checkTerm_DcylFollower(__VA_ARGS__)
#define sendInitC(...) sendInitC_DcylFollower(__VA_ARGS__)
#define setRefFrm()    setRefFrm_DCylFollower()
//TODO:
// - 2/N fish want open bc in z
// - cleaning: maybe compile cubism and set flags based on user's app choice
#endif

#include "../Definitions.h"
#include "../ObstacleBlock.h"

CubismUP_3D_NAMESPACE_BEGIN

struct StateReward
{
  double lengthscale, timescale;
  double velscale   = lengthscale/timescale;
  double forcescale = velscale*velscale*lengthscale*lengthscale; //l^4/t^2
  double powerscale = forcescale*velscale; //rho*l^3 * l/t^2 * l/t

  bool bRestart = false;
  bool bForgiving=0, bLoadedActions=0, bInteractive=0, randomStart=0;
  //bool randomActions, bSpiral;
  int info=1, stepId=0;//, nActions=2;
  double t_next_comm=0, Tstartlearn=1e9, GoalDX=0, new_curv=0, old_curv=0, new_Tp=0;

  //exponential averages
  double thExp = 0, vxExp = 0, vyExp = 0, avExp = 0;
  //average quantities
  double avg_wght = 0;
  double ThetaAvg = 0, ThetaVel = 0, VxAvg = 0, VyAvg = 0, AvAvg = 0;
  double PoutBnd = 0, Pout = 0, defPowerBnd = 0, defPower = 0, ToD = 0;
  double EffPDefBnd = 0, EffPDef = 0, Pthrust = 0, Pdrag = 0;
  void resetAverage()
  {
    avg_wght = ThetaAvg = ThetaVel = VxAvg = VyAvg = AvAvg = Pthrust = ToD = 0;
    PoutBnd = Pout = defPowerBnd = defPower = Pdrag = EffPDefBnd = EffPDef = 0;
  }
  void updateAverages(const double _dt,
   const double _th,   const double _vx,   const double _vy,  const double _av,
   const double _pO1,  const double _pO2,  const double _pW1, const double _pW2,
   const double _eff1, const double _eff2, const double _pT,  const double _pD,
   const double _T,    const double _D)
  {
    if(_dt<=0) return;

    const double _ToD=_D<1e-9?0:_T/_D, _W=1/(avg_wght+_dt), _vt=atan2(_vy,_vx);

    VxAvg       = (       VxAvg * avg_wght + _vx   * _dt ) * _W;
    VyAvg       = (       VyAvg * avg_wght + _vy   * _dt ) * _W;
    AvAvg       = (       AvAvg * avg_wght + _av   * _dt ) * _W;
    ThetaAvg    = (    ThetaAvg * avg_wght + _th   * _dt ) * _W;
    ThetaVel    = (    ThetaVel * avg_wght + _vt   * _dt ) * _W;
    Pout        = (        Pout * avg_wght + _pO1  * _dt ) * _W;
    PoutBnd     = (     PoutBnd * avg_wght + _pO2  * _dt ) * _W;
    defPower    = (    defPower * avg_wght + _pW1  * _dt ) * _W;
    defPowerBnd = ( defPowerBnd * avg_wght + _pW2  * _dt ) * _W;
    EffPDef     = (     EffPDef * avg_wght + _eff1 * _dt ) * _W;
    EffPDefBnd  = (  EffPDefBnd * avg_wght + _eff2 * _dt ) * _W;
    Pthrust     = (     Pthrust * avg_wght + _pT   * _dt ) * _W;
    Pdrag       = (       Pdrag * avg_wght + _pD   * _dt ) * _W;
    ToD         = (         ToD * avg_wght + _ToD  * _dt ) * _W;

    avg_wght   += _dt;
    battery    += _dt * defPowerBnd;
    thExp = (1-_dt) * thExp + _dt * _th;
    vxExp = (1-_dt) * vxExp + _dt * _vx;
    vyExp = (1-_dt) * vyExp + _dt * _vy;
    avExp = (1-_dt) * avExp + _dt * _av;
  }

  //inst quantitites
  double Xrel = 0, Xabs = 0, Xpov = 0, Yrel = 0, Yabs = 0, Ypov = 0, Theta = 0;
  double VxInst = 0, VyInst = 0, AvInst = 0, VX = 0, VY = 0, AV = 0;
  double phaseShift = 0, Dist = 0, Quad = 0, RelAng = 0;
  double battery = 1, ext_X = -1, ext_Y = -1, ext_Z = -1;
  void updateInstant(
    const double _xR, const double _xA, const double _yR, const double _yA,
    const double _th, const double _vx, const double _vy, const double _av)
  {
      Xrel = _xR; Xabs = _xA; Yrel = _yR; Yabs = _yA; Theta= _th;
      VxInst=_vx; VyInst=_vy; AvInst=_av;
      if (Xrel<0.05 || Yrel<0.025)    bRestart = true;
      if (ext_X>0 && ext_X-Xrel<0.2)  bRestart = true;
      if (ext_Y>0 && ext_Y-Yrel<.025) bRestart = true;
  }

  //sensors
  vector<double> FPAbove, FVAbove, FPBelow, FVBelow;
  vector<double> PXAbove, PYAbove, PXBelow, PYBelow;
  vector<double> raySight;
  vector<vector<double>> loadedActions;

  StateReward(const double _lengthscale = 1, const double _timescale = 1) :
  lengthscale(_lengthscale), timescale(_timescale)
  {
    //printf("scales: %f %f %f %f %f",
    //  lengthscale,timescale,velscale,forcescale,powerscale);
    FPAbove.resize(NpLatLine,0); FVAbove.resize(NpLatLine,0);
    FPBelow.resize(NpLatLine,0); FVBelow.resize(NpLatLine,0);
    PXAbove.resize(NpLatLine,0); PYAbove.resize(NpLatLine,0);
    PXBelow.resize(NpLatLine,0); PYBelow.resize(NpLatLine,0);
    raySight.resize(2*NpLatLine,0);
  }
  StateReward& operator= (const StateReward& s)
  {
    lengthscale = s.lengthscale;
    timescale   = s.timescale;
    velscale    = lengthscale/timescale;
    forcescale  = velscale*velscale*lengthscale*lengthscale; //l^4/t^2
    powerscale  = forcescale*velscale; //rho*l^3 * l/t^2 * l/t

    #ifdef __RL_TRAINING
    printf("scales: %f %f %f %f %f",
      lengthscale,timescale,velscale,forcescale,powerscale);
    #endif
    FPAbove.resize(NpLatLine,0); FVAbove.resize(NpLatLine,0);
    FPBelow.resize(NpLatLine,0); FVBelow.resize(NpLatLine,0);
    PXAbove.resize(NpLatLine,0); PYAbove.resize(NpLatLine,0);
    PXBelow.resize(NpLatLine,0); PYBelow.resize(NpLatLine,0);
    raySight.resize(2*NpLatLine,0);
    return *this;
  }

  void parseArguments(ArgumentParser & parser)
  {
    bInteractive = parser("-interactive").asBool(false);
    Tstartlearn = parser("-Tstartlearn").asDouble(bInteractive ? timescale : 1e9);
    GoalDX = parser("-GoalDX").asDouble(0);
    //nActions = parser("-nActions").asInt(2);
    bForgiving = parser("-easyFailBox").asBool(false);
    randomStart = parser("-randomStart").asBool(false);
    bLoadedActions = parser("-useLoadedActions").asBool(false);
    //hardcoded to compute avg state components for halfT b4 first comm... iffy
    t_next_comm = Tstartlearn;// - timescale/2;
    if (bLoadedActions) readLoadedActions();
    printf("scales: %f %f %f %f %f, %d, %f, %f, %d, %d, %d\n",
      lengthscale,timescale,velscale,forcescale,powerscale, bInteractive, Tstartlearn, GoalDX, bForgiving, randomStart, bLoadedActions);
  }

  vector<double> useLoadedActions()
  {
    if (loadedActions.size()>1) {
        vector<double> actions = loadedActions.back();
        loadedActions.pop_back();
        return actions;
    } //else zero actions
    else return vector<double>();
  }

  void readLoadedActions(const int nActions = 2)
  {
    double dummy_time;
    vector<double> action(nActions);
    ifstream in("orders_1.txt");
    std::string line;
    if(in.good()) {
      while (getline(in, line)) {
        std::istringstream line_in(line);
        if(nActions==2) line_in >> dummy_time >> action[0] >> action[1];
        else line_in >> dummy_time >> action[0];
        //i want to do pop back later:
        loadedActions.insert(loadedActions.begin(),action);
      }
      in.close();
    }
  }

  void updateStepId(const int _stepId) {stepId=_stepId;}

  void finalize(const double  xFOR,   const double yFOR, const double thFOR,
                const double vxFOR,  const double vyFOR, const double avFOR)
  {
    //velocity of reference from fish pov
    VX = (VxInst-vxFOR)*std::cos(Theta) + (VyInst-vyFOR)*std::sin(Theta);
    VY = (VyInst-vyFOR)*std::cos(Theta) - (VxInst-vxFOR)*std::sin(Theta);
    AV = (AvInst-avFOR);
    //velocity of fish in reference pov
    const double vxAvg = VxAvg, vyAvg = VyAvg;
    VxAvg = vxAvg*std::cos(Theta) + vyAvg*std::sin(Theta);
    VyAvg = vyAvg*std::cos(Theta) - vxAvg*std::sin(Theta);
    AvAvg = AvAvg;
    //position in reference frame
    Xpov = (Xrel-xFOR)*std::cos(thFOR) + (Yrel-yFOR)*std::sin(thFOR);
    Ypov = (Yrel-yFOR)*std::cos(thFOR) - (Xrel-xFOR)*std::sin(thFOR);
    RelAng = Theta - thFOR;

    const double Xframe=(xFOR-Xrel)*std::cos(Theta)+(yFOR-Yrel)*std::sin(Theta);
    const double Yframe=(yFOR-Yrel)*std::cos(Theta)-(xFOR-Xrel)*std::sin(Theta);
    Dist = std::sqrt(std::pow(Xrel-xFOR,2) + std::pow(Yrel-yFOR,2));
    Quad = std::atan2(Yframe, Xframe);
  }

  bool checkTerm_LeadFollower(const double xFOR, const double yFOR,
    const double thFOR,const double vxFOR,const double vyFOR,const double avFOR)
  {
    checkTerm_bounds(xFOR, yFOR);
    if(not bInteractive or bRestart) return bRestart;
    const double _Xrel = (Xrel-xFOR)*cos(thFOR) + (Yrel-yFOR)*sin(thFOR);
    const double _Yrel = (Yrel-yFOR)*cos(thFOR) - (Xrel-xFOR)*sin(thFOR);
    const double _thRel= Theta - thFOR;
    const double _Dist = sqrt(pow(Xrel-xFOR,2) + pow(Yrel-yFOR,2));

    bRestart = _Dist < .25*lengthscale;
    if(bRestart) {printf("Too close\n"); return bRestart;}
    //at DX=1, allowed DY=.5, at DX=2.5 allowed DY=.75
    bRestart = fabs(_Yrel)>(bForgiving?lengthscale: _Xrel/6 + 7*lengthscale/12);
    if(bRestart) {printf("Too much vertical distance\n"); return bRestart;}
    #ifdef __ExploreHalfWake
      bRestart = _Yrel < -.1*lengthscale;
      if(bRestart) {printf("Wrong half of the wake\n"); return bRestart;}
    #endif
    bRestart = std::fabs(_thRel)> (bForgiving ? M_PI : M_PI/2);
    if(bRestart) {printf("Too different inclination\n"); return bRestart;}
    bRestart = _Xrel < lengthscale || _Xrel > 2.5*lengthscale;
    if(bRestart) {printf("Too far from horizontal goal\n"); return bRestart;}
    return bRestart;
  }
  bool checkTerm_DcylFollower(const double xFOR,const double yFOR,
    const double thFOR,const double vxFOR,const double vyFOR,const double avFOR)
  {
    if (bRestart) printf("Already ended\n");
    if(not bInteractive or bRestart) return bRestart;
    for(int i=0; i<NpLatLine; i++) {
      if(PXAbove[i]< xFOR||PXBelow[i]< xFOR) {printf("Touching\n"); bRestart=1;}
      if(PXAbove[i]>  0.8||PXBelow[i]>  0.8) {printf("Boundary\n"); bRestart=1;}
      if(PYAbove[i]<    0||PYBelow[i]<    0) {printf("Boundary\n"); bRestart=1;}
      if(PYAbove[i]>ext_Y||PYBelow[i]>ext_Y) {printf("Boundary\n"); bRestart=1;}
      if(bRestart) return bRestart;
    }
    const double _Xrel = (Xrel-xFOR)*cos(thFOR) + (Yrel-yFOR)*sin(thFOR);
    const double _Yrel = (Yrel-yFOR)*cos(thFOR) - (Xrel-xFOR)*sin(thFOR);
    const double _Dist = sqrt(pow(Xrel-xFOR,2) + pow(Yrel-yFOR,2));
    (void)_Xrel;  // To stop complaining about unused variables.
    (void)_Yrel;
    (void)_Dist;
    bRestart = std::fabs(_Yrel) > 2*lengthscale;
    if(bRestart) {printf("Too much vertical distance\n"); return bRestart;}
    bRestart = std::fabs(Theta)>M_PI;
    if(bRestart) {printf("Too different inclination\n"); return bRestart;}
    return bRestart;
  }
  bool checkTerm_bounds(const double xFOR, const double yFOR)
  {
    if ( Xrel<.05*lengthscale || Yrel<.025*lengthscale) bRestart = true;
    if ( ext_X>0 && ext_X-Xrel < .2  *lengthscale ) bRestart = true;
    if ( ext_Y>0 && ext_Y-Yrel < .025*lengthscale ) bRestart = true;
    if (bRestart) printf("Out of bounds\n");
    return bRestart;
  }

  struct skinForcesVels
  {
    skinForcesVels(const int _nDest) : nDest(_nDest), data(_alloc(5*_nDest))
    {
      memset(data, 0, sizeof(double)*5*nDest);
    }

    virtual ~skinForcesVels() { _dealloc(data); }

    inline void storeNearest(const double fxP, const double fyP, const double fxV, const double fyV, const int i)
    {
      data[i+0*nDest] += fxP; data[i+1*nDest] += fyP;
      data[i+2*nDest] += fxV; data[i+3*nDest] += fyV;
      data[i+4*nDest] += 1.;
    }

    inline double fxP(const int i) { return data[i+0*nDest]; }
    inline double fyP(const int i) { return data[i+1*nDest]; }
    inline double fxV(const int i) { return data[i+2*nDest]; }
    inline double fyV(const int i) { return data[i+3*nDest]; }

    void synchronize(const MPI_Comm comm)
    {
      //int rank;
      //MPI_Comm_rank(comm, &rank);
      #ifndef CUP_SINGLE_PRECISION
      MPI_Allreduce(MPI_IN_PLACE, data, 5*nDest, MPI_DOUBLE, MPI_SUM, comm);
      #else //CUP_SINGLE_PRECISION
      MPI_Allreduce(MPI_IN_PLACE, data, 5*nDest, MPI_FLOAT,  MPI_SUM, comm);
      #endif//
    }
    void print(const MPI_Comm comm, const int stepNumber)
    {
      int rank;
      MPI_Comm_rank(comm, &rank);
      if(rank) return;
      ofstream fout;
      char buf[500];
      sprintf(buf, "midplaneData_%07d.txt", stepNumber);
      string filename(buf);
      fout.open(filename, ios::trunc);
      for(int i=0; i<nDest; ++i)
      fout<<fxP(i)<<"\t"<<fyP(i)<<"\t"<<fxV(i)<<"\t"<<fyV(i)<<"\t"<<std::endl;
      fout.close();
    }

    private:
    const int nDest;
    double*const data;
    double * _alloc(const int N) { return new double[N]; }
    void _dealloc(double * ptr) {
      if(ptr not_eq nullptr) {
          delete [] ptr;
          ptr=nullptr;
      }
    }
  };

  typedef const Real*const constAry;
  void nearestGridPoints(
    const std::map<int,ObstacleBlock*>& obstacleBlocks,
    const vector<BlockInfo>& vInfo, const int Nskin,
    constAry  xU, constAry  yU, constAry  xL, constAry yL,
    constAry nxU, constAry nyU, constAry nxL, constAry nyL,
    const double zObst, const double h, const MPI_Comm comm)
  {
    constexpr int BS = FluidBlock::BS;
    skinForcesVels data(Nskin*2);
    const double eps = 10*std::numeric_limits<double>::epsilon();
    const unsigned NB = vInfo.size();

    #pragma omp parallel for schedule(dynamic)
    for (int j=0; j<2*Nskin; j++)
    {
      const double X = j>=Nskin ? xL[j-Nskin] : xU[j];
      const double Y = j>=Nskin ? yL[j-Nskin] : yU[j];

        for(unsigned i=0; i<NB; i++) {
          const BlockInfo I = vInfo[i];
          const auto pos = obstacleBlocks.find(I.blockID);
          if(pos == obstacleBlocks.end()) continue;
          if(pos->second->nPoints == 0) continue;
          const auto& o = pos->second;
          assert(o->filled);
          double max_pos[3], min_pos[3];
          I.pos(min_pos, 0, 0, 0);
          I.pos(max_pos, BS-1, BS-1, BS-1);
          if(zObst-max_pos[2]>h+eps || min_pos[2]-zObst>h+eps) continue;
          if(Y    -max_pos[1]>h+eps || min_pos[1]-Y    >h+eps) continue;
          if(X    -max_pos[0]>h+eps || min_pos[0]-X    >h+eps) continue;

          for(int k=0; k<pos->second->nPoints; k++) {
            if(std::fabs(o->pZ[k]-zObst)>h+eps) continue;
            if(std::fabs(o->pY[k]-Y)    >h+eps) continue;
            if(std::fabs(o->pX[k]-X)    >h+eps) continue;
            //printf("%f %f %f %f\n",o->fxP[k],o->fyP[k],o->fxV[k],o->fyV[k]);
            data.storeNearest(o->fxP[k], o->fyP[k], o->fxV[k], o->fyV[k], j);
          }
        }
    }

    data.synchronize(comm);
    //data.print(comm,stepId);

    /*
      int rank;
      MPI_Comm_rank(comm, &rank);
      if(!rank) {
        ofstream fileskin;
        char buf[500];
        sprintf(buf, "skinPoints_%07d.txt", stepId);
        string filename(buf);
        fileskin.open(filename, ios::trunc);
        for (int j=0;       j<Nskin; j++)
            fileskin<<xU[j]<<"\t"<<yU[j]<<std::endl;
        for (int j=Nskin-1; j>=0;    j--)
            fileskin<<xL[j]<<"\t"<<yL[j]<<std::endl;
        fileskin.close();
      }
    */

    vector<double> NxAbove(NpLatLine,0), NyAbove(NpLatLine,0);
    vector<double> NxBelow(NpLatLine,0), NyBelow(NpLatLine,0);
    //now, feed the sensors
    for (int k=0; k<NpLatLine; k++)
    {
        const int first = k   *(double)Nskin/(double)NpLatLine;
        const int last = (k+1)*(double)Nskin/(double)NpLatLine;
        double  FPxAbove=0, FPxBelow=0, FPyAbove=0, FPyBelow=0;
        double  FVxAbove=0, FVxBelow=0, FVyAbove=0, FVyBelow=0;

        for (int j=first; j<last; j++) {
            FPxAbove += data.fxP(j); FPxBelow += data.fxP(j+Nskin);
            FPyAbove += data.fyP(j); FPyBelow += data.fyP(j+Nskin);
            FVxAbove += data.fxV(j); FVxBelow += data.fxV(j+Nskin);
            FVyAbove += data.fyV(j); FVyBelow += data.fyV(j+Nskin);
        }

        const int mid = 0.5*(first+last);
        PXAbove[k] = xU[mid]; PYAbove[k] = yU[mid];
        PXBelow[k] = xL[mid]; PYBelow[k] = yL[mid];

        const double nxAbove = nxU[mid]; // ^            ^
        const double nyAbove = nyU[mid]; //   `        /
        const double txAbove = nyU[mid]; //   n `    / t
        const double tyAbove =-nxU[mid]; //       `
        NxAbove[k] = nxAbove;
        NyAbove[k] = nyAbove;
        const double nxBelow = nxL[mid]; //    /`
        const double nyBelow = nyL[mid]; // n /    `  t
        const double txBelow =-nyL[mid]; //  /        `
        const double tyBelow = nxL[mid]; // v            v
        NxBelow[k] = nxBelow;
        NyBelow[k] = nyBelow;

        FPAbove[k] = FPxAbove*nxAbove + FPyAbove*nyAbove;
        FVAbove[k] = FVxAbove*txAbove + FVyAbove*tyAbove;
        FPBelow[k] = FPxBelow*nxBelow + FPyBelow*nyBelow;
        FVBelow[k] = FVxBelow*txBelow + FVyBelow*tyBelow;
    }
    if(0){
        ofstream fileskin;
        char buf[500];
        sprintf(buf, "sensorDistrib_%07d.txt", stepId);
        string filename(buf);
        fileskin.open(filename, ios::trunc);
        // int k=0;
        for(int i=0; i<NpLatLine; ++i)
            fileskin<<PXAbove[i]<<"\t"<<PYAbove[i]<<"\t"<<NxAbove[i]
              <<"\t"<<NyAbove[i]<<"\t"<<FPAbove[i]<<"\t"<<FVAbove[i]
              //<<"\t"<<raySight[k++]
              <<std::endl;
        for(int i=0; i<NpLatLine; ++i)
            fileskin<<PXBelow[i]<<"\t"<<PYBelow[i]<<"\t"<<NxBelow[i]
              <<"\t"<<NyBelow[i]<<"\t"<<FPBelow[i]<<"\t"<<FVBelow[i]
              //<<"\t"<<raySight[k++]
              <<std::endl;
        fileskin.close();
    }
  }

  void save(const int step_id, string filename)
  {
    ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<double>::digits10 + 1);
    string fullFileName = filename==string() ? "restart_IF2D_Stefan" : filename;

    savestream.open(fullFileName+"_save_data.txt");

    savestream << bRestart << "\t" << info << "\t" << avg_wght << "\t" << t_next_comm << "\t"
      << Xrel << "\t" << Xabs << "\t" << Yrel << "\t" << Yabs << "\t"
      << Theta << "\t" << VxAvg << "\t" << VyAvg<< "\t" << AvAvg << "\t"
      << thExp << "\t" << vxExp << "\t" << vyExp<< "\t" << avExp << "\t"
      << VxInst << "\t" << VyInst<< "\t" << AvInst << "\t"
      << Dist << "\t" << Quad << "\t" << RelAng<< "\t"
      << VX << "\t" << VY << "\t" << AV << "\t"
      << ThetaAvg<<"\t"<<ThetaVel<<"\t"<<PoutBnd<<"\t"<<Pout << "\t"
      << defPowerBnd<<"\t"<<defPower<<"\t"<<EffPDefBnd<<"\t"<<EffPDef << "\t"
      << Pthrust << "\t" << Pdrag << "\t" << ToD << std::endl;

    for (int i=0; i<NpLatLine; i++) {
      savestream <<
      PXAbove[i] << "\t" << PYAbove[i] << "\t" <<
      PXBelow[i] << "\t" << PYBelow[i] << "\t" <<
      FPAbove[i] << "\t" << FVAbove[i] << "\t" <<
      FPBelow[i] << "\t" << FVBelow[i] << std::endl;
    }

    savestream.close();
  }

  void restart(string filename)
  {
    ifstream restartstream;
    string fullFileName = filename;
    restartstream.open(fullFileName+"_save_data.txt");
    if(not restartstream.good()) return;

    restartstream >> bRestart >> info >> avg_wght >> t_next_comm >>
    Xrel >> Xabs >> Yrel >> Yabs >>
    Theta >> VxAvg >> VyAvg >> AvAvg >>
    thExp >> vxExp >> vyExp >> avExp >>
    VxInst >> VyInst >> AvInst >>
    Dist >> Quad >> RelAng >>
    VX >> VY >> AV >>
    ThetaAvg >> ThetaVel >> PoutBnd >> Pout >>
    defPowerBnd >> defPower >> EffPDefBnd>> EffPDef >>
    Pthrust >> Pdrag >> ToD;

    for (int i=0; i<NpLatLine; i++) {
        restartstream >>
        PXAbove[i] >> PYAbove[i] >>
        PXBelow[i] >> PYBelow[i] >>
        FPAbove[i] >> FVAbove[i] >>
        FPBelow[i] >> FVBelow[i];
    }

    restartstream.close();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank==0)
    {
      cout << bRestart << "\t" << info << "\t" << avg_wght << "\t" << t_next_comm << "\t"
      << Xrel << "\t" << Xabs << "\t" << Yrel << "\t" << Yabs << "\t"
      << Theta << "\t" << VxAvg << "\t" << VyAvg<< "\t" << AvAvg << "\t"
      << thExp << "\t" << vxExp << "\t" << vyExp<< "\t" << avExp << "\t"
      << VxInst << "\t" << VyInst<< "\t" << AvInst << "\t"
      << Dist << "\t" << Quad << "\t" << RelAng<< "\t"
      << VX << "\t" << VY << "\t" << AV << "\t"
      << ThetaAvg << "\t" << ThetaVel << "\t" << PoutBnd << "\t" << Pout << "\t"
      << defPowerBnd << "\t" << defPower << "\t" << EffPDefBnd<< "\t" << EffPDef << "\t"
      << Pthrust << "\t" << Pdrag << "\t" << ToD << std::endl;

      for (int i=0; i<NpLatLine; i++) {
      cout << PXAbove[i] << "\t" << PYAbove[i] << "\t" <<
              PXBelow[i] << "\t" << PYBelow[i] << "\t" <<
              FPAbove[i] << "\t" << FVAbove[i] << "\t" <<
              FPBelow[i] << "\t" << FVBelow[i] << std::endl;
      }
    }
  }

  void print(const int ID, const int stepNumber, const double time)
  {
    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    //if (rank) return;
    {
      ofstream fileskin;
      char buf[500];
      sprintf(buf, "sensorDistrib_%1d_%07d.txt", ID, stepNumber);
      string filename(buf);
      fileskin.open(filename, ios::trunc);
      int k=0;
      for(int i=0; i<NpLatLine; ++i)
          fileskin<<PXAbove[i]<<"\t"<<PYAbove[i]<<"\t"<<FPAbove[i]<<"\t"<<FVAbove[i]<<"\t"<<raySight[k++]<<std::endl;
      for(int i=0; i<NpLatLine; ++i)
          fileskin<<PXBelow[i]<<"\t"<<PYBelow[i]<<"\t"<<FPBelow[i]<<"\t"<<FVBelow[i]<<"\t"<<raySight[k++]<<std::endl;
      fileskin.close();
    }
    {
      ofstream fileskin;
      char buf[500];
      sprintf(buf, "avgSensors_%1d.txt",ID);
      string filename(buf);
      fileskin.open(filename, ios::app);

      fileskin<< avg_wght << "\t" << t_next_comm  << "\t"
              << Xrel << "\t" << Xabs << "\t" << Yrel << "\t" << Yabs << "\t"
              << Theta << "\t" << VxAvg << "\t" << VyAvg<< "\t" << AvAvg << "\t"
              << thExp << "\t" << vxExp << "\t" << vyExp<< "\t" << avExp << "\t"
              << VxInst << "\t" << VyInst<< "\t" << AvInst << "\t"
              << Dist << "\t" << Quad << "\t" << RelAng<< "\t"
              << VX << "\t" << VY << "\t" << AV << "\t"
              << ThetaAvg << "\t" << ThetaVel << "\t" << PoutBnd << "\t" << Pout << "\t"
              << defPowerBnd << "\t" << defPower << "\t" << EffPDefBnd<< "\t" << EffPDef << "\t"
              << Pthrust << "\t" << Pdrag << "\t" << ToD << std::endl;
      fileskin.close();
    }
  }

  vector<double> fillState(const double time, const int nStateVars, const int nActions = 2)
  {
    if(bRestart) {
      if(info==1) printf("Reached termination before first action!!!\n");
      info = 2;
    }
    vector<double> state(nStateVars, 0);
    int k = 0;
    //state[k++] = sr.Xpov*invlscale - GoalDX;
    state[k++] = Xpov / lengthscale;
    state[k++] = Ypov / lengthscale;
    state[k++] = RelAng;
    state[k++] = std::fmod(time, timescale); //1 is Tperiod of leader
    state[k++] = new_curv;
    state[k++] = old_curv;

    if(nActions==2)
    {
    state[k++] = new_Tp;
    state[k++] = phaseShift;
    state[k++] = VX / velscale;
    state[k++] = VY / velscale;
    state[k++] = AV / velscale;
    }

    state[k++] =        Dist / lengthscale;
    state[k++] =       Quad;
    state[k++] =       VxAvg / velscale;
    state[k++] =       VyAvg / velscale;
    state[k++] =       AvAvg / velscale;
    state[k++] =        Pout / powerscale;
    state[k++] =    defPower / powerscale;
    state[k++] =     EffPDef;
    state[k++] =     PoutBnd / powerscale;
    state[k++] = defPowerBnd / powerscale;
    state[k++] =  EffPDefBnd;
    state[k++] =     Pthrust / powerscale;
    state[k++] =       Pdrag / powerscale;
    state[k++] =         ToD;

    if(nStateVars>=k+4*NpLatLine) {
      for (int j=0; j<NpLatLine; j++) state[k++] =  FPAbove[j] / forcescale;
      //for (int j=0; j<NpLatLine; j++) printf("FPAbove %d %f\n",j,FPAbove[j]);
      for (int j=0; j<NpLatLine; j++) state[k++] =  FVAbove[j] / forcescale;
      //for (int j=0; j<NpLatLine; j++) printf("FVAbove %d %f\n",j,FVAbove[j]);
      for (int j=0; j<NpLatLine; j++) state[k++] =  FPBelow[j] / forcescale;
      //for (int j=0; j<NpLatLine; j++) printf("FPBelow %d %f\n",j,FPBelow[j]);
      for (int j=0; j<NpLatLine; j++) state[k++] =  FVBelow[j] / forcescale;
      //for (int j=0; j<NpLatLine; j++) printf("FVBelow %d %f\n",j,FVBelow[j]);
    }
    if(nStateVars>=k+2*NpLatLine)
      for (int j=0;j<2*NpLatLine;j++) state[k++] = raySight[j] / lengthscale;

    return state;
  }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_StateRewardData_h
