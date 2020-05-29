//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_ObstacleBlock_h
#define CubismUP_3D_ObstacleBlock_h

#include "Definitions.h"

// to shift the surface where I compute gradchi for surface integrals
// if set to value greater than 0, it shifts surface by that many mesh sizes
#define SURFDH 1
#include <vector> //surface vector
#include <cstring> //memset
#include <cstdio> //print

CubismUP_3D_NAMESPACE_BEGIN

struct surface_data
{
  const int ix, iy, iz;
  const double dchidx, dchidy, dchidz, delta;

  surface_data(int _ix, int _iy, int _iz, double _dchidx, double _dchidy,
    double _dchidz, double _delta) : ix(_ix), iy(_iy), iz(_iz),
    dchidx(_dchidx), dchidy(_dchidy), dchidz(_dchidz), delta(_delta) {}
  surface_data() = delete;
};

struct ObstacleBlock
{
  static constexpr int BS = FluidBlock::BS;
  static constexpr int sizeX = BS;
  static constexpr int sizeY = BS;
  static constexpr int sizeZ = BS;

  // bulk quantities:
  Real          chi[BS][BS][BS];
  Real          sdf[BS][BS][BS];
  Real         udef[BS][BS][BS][3];
  int sectionMarker[BS][BS][BS];

  //surface quantities:
  int nPoints = 0;
  bool filled = false;
  std::vector<surface_data*> surface;
  int *ss  = nullptr;
  double *pX  = nullptr, *pY  = nullptr, *pZ  = nullptr, *P = nullptr;
  double *fX  = nullptr, *fY  = nullptr, *fZ  = nullptr;
  double *fxP = nullptr, *fyP = nullptr, *fzP = nullptr;
  double *fxV = nullptr, *fyV = nullptr, *fzV = nullptr;
  double *vX  = nullptr, *vY  = nullptr, *vZ  = nullptr;
  double *vxDef = nullptr, *vyDef = nullptr, *vzDef = nullptr;

  //additive quantities:
  // construct CHI & center of mass interpolated on grid:
  double CoM_x = 0, CoM_y = 0, CoM_z = 0, mass = 0;

  // Penalization volume integral forces:
  double V =0, FX=0, FY=0, FZ=0, TX=0, TY=0, TZ=0;
  double J0=0, J1=0, J2=0, J3=0, J4=0, J5=0;

  double GfX=0, GpX=0, GpY=0, GpZ=0, Gj0=0, Gj1=0, Gj2=0, Gj3=0, Gj4=0, Gj5=0;
  double GuX=0, GuY=0, GuZ=0, GaX=0, GaY=0, GaZ=0;

  // Fluid-structure interface forces: (these are the 22 quantities of nQoI)
  double  forcex   = 0,  forcey   = 0,  forcez   = 0;
  double  forcex_P = 0,  forcey_P = 0,  forcez_P = 0;
  double  forcex_V = 0,  forcey_V = 0,  forcez_V = 0;
  double torquex   = 0, torquey   = 0, torquez   = 0;
  double  gammax   = 0,  gammay   = 0,  gammaz   = 0;
  //Real torquex_P = 0, torquey_P = 0, torquez_P = 0;
  //Real torquex_V = 0, torquey_V = 0, torquez_V = 0;
  double drag = 0, thrust = 0, Pout=0, PoutBnd=0, defPower=0, defPowerBnd = 0, pLocom = 0;
  static const int nQoI = 22;

  virtual void sumQoI(std::vector<double>& sum)
  {
    assert(sum.size() == nQoI);
    unsigned k = 0;
    sum[k++] += forcex;   sum[k++] += forcey;   sum[k++] += forcez;
    sum[k++] += forcex_P; sum[k++] += forcey_P; sum[k++] += forcez_P;
    sum[k++] += forcex_V; sum[k++] += forcey_V; sum[k++] += forcez_V;
    sum[k++] += torquex;  sum[k++] += torquey;  sum[k++] += torquez;
    sum[k++] += gammax;   sum[k++] += gammay;   sum[k++] += gammaz;
    sum[k++] += drag;     sum[k++] += thrust;   sum[k++] += Pout;
    sum[k++] += PoutBnd;  sum[k++] += defPower; sum[k++] += defPowerBnd;
    sum[k++] += pLocom;
  }

  ObstacleBlock()
  {
    //rough estimate of surface cutting the block diagonally
    //with 2 points needed on each side of surface
    surface.reserve(4*BS);
  }
  virtual ~ObstacleBlock()
  {
    clear_surface();
  }

  void clear_surface()
  {
    filled=false;
    nPoints=0;
    CoM_x    = CoM_y    = CoM_z    =0;
    forcex   = forcey   = forcez   =0;
    forcex_P = forcey_P = forcez_P =0;
    forcex_V = forcey_V = forcez_V =0;
    gammax   = gammay   = gammaz   =0;
    torquex  = torquey  = torquez  =0;
    //torquex_P=torquey_P=torquez_P=0;
    //torquex_V=torquey_V=torquez_V=0;
    mass=drag=thrust=Pout=PoutBnd=defPower=defPowerBnd=0;

    for (auto & trash : surface) {
      if(trash == nullptr) continue;
      delete trash;
      trash = nullptr;
    }
    surface.clear();
    if(pX      not_eq nullptr){free(pX);      pX=nullptr;     }
    if(pY      not_eq nullptr){free(pY);      pY=nullptr;     }
    if(pZ      not_eq nullptr){free(pZ);      pZ=nullptr;     }
    if(P       not_eq nullptr){free(P);       P=nullptr;      }
    if(fX      not_eq nullptr){free(fX);      fX=nullptr;     }
    if(fY      not_eq nullptr){free(fY);      fY=nullptr;     }
    if(fZ      not_eq nullptr){free(fZ);      fZ=nullptr;     }
    if(fxP     not_eq nullptr){free(fxP);     fxP=nullptr;    }
    if(fyP     not_eq nullptr){free(fyP);     fyP=nullptr;    }
    if(fzP     not_eq nullptr){free(fzP);     fzP=nullptr;    }
    if(fxV     not_eq nullptr){free(fxV);     fxV=nullptr;    }
    if(fyV     not_eq nullptr){free(fyV);     fyV=nullptr;    }
    if(fzV     not_eq nullptr){free(fzV);     fzV=nullptr;    }
    if(vX      not_eq nullptr){free(vX);      vX=nullptr;     }
    if(vY      not_eq nullptr){free(vY);      vY=nullptr;     }
    if(vZ      not_eq nullptr){free(vZ);      vZ=nullptr;     }
    if(vxDef   not_eq nullptr){free(vxDef);   vxDef=nullptr;  }
    if(vyDef   not_eq nullptr){free(vyDef);   vyDef=nullptr;  }
    if(vzDef   not_eq nullptr){free(vzDef);   vzDef=nullptr;  }
    if(ss      not_eq nullptr){free(ss);      ss=nullptr;     }
  }

  virtual void clear()
  {
    clear_surface();
    memset(chi,  0, sizeof(Real)*sizeX*sizeY*sizeZ);
    memset(sdf,  0, sizeof(Real)*sizeX*sizeY*sizeZ);
    memset(udef, 0, sizeof(Real)*sizeX*sizeY*sizeZ*3);
    memset(sectionMarker, 0, sizeof(int)*sizeX*sizeY*sizeZ);
  }

  inline void write(const int ix, const int iy, const int iz, const Real delta, const Real gradUX, const Real gradUY, const Real gradUZ)
  {
    //if(_chi<chi[iz][iy][ix]) return;
    assert(!filled);
    nPoints++;
    const double dchidx = -delta*gradUX;
    const double dchidy = -delta*gradUY;
    const double dchidz = -delta*gradUZ;
    surface.push_back(new surface_data(ix,iy,iz,dchidx,dchidy,dchidz,delta));
  }

  void allocate_surface()
  {
    filled = true;
    assert((int)surface.size() == nPoints);
    assert(pX==nullptr && pY==nullptr && pZ==nullptr);
    assert(vX==nullptr && vY==nullptr && vZ==nullptr);
    assert(fX==nullptr && fY==nullptr && fZ==nullptr);
    pX   =init<double>(nPoints); pY   =init<double>(nPoints);
    pZ   =init<double>(nPoints); vX   =init<double>(nPoints);
    vY   =init<double>(nPoints); vZ   =init<double>(nPoints);
    fX   =init<double>(nPoints); fY   =init<double>(nPoints);
    fZ   =init<double>(nPoints); fxP  =init<double>(nPoints);
    fyP  =init<double>(nPoints); fzP  =init<double>(nPoints);
    fxV  =init<double>(nPoints); fyV  =init<double>(nPoints);
    fzV  =init<double>(nPoints); vxDef=init<double>(nPoints);
    vyDef=init<double>(nPoints); vzDef=init<double>(nPoints);
    P    =init<double>(nPoints); ss   =init<int>(nPoints);
  }

  template <typename T>
  static inline T* init(const int N)
  {
    T* ptr;
    const int ret = posix_memalign((void **)&ptr, 32, N * sizeof(T));
    if (ret == EINVAL) {
        fprintf(stderr, "posix_memalign somehow returned EINVAL...\n");
        fflush(0); abort();
    } else if (ret == ENOMEM) {
        fprintf(stderr, "Cannot allocate %dx%d bytes with align 32!\n",
                N, (int)sizeof(T));
        fflush(0); abort();
    }
    assert(ptr != nullptr);
    memset(ptr, 0, N * sizeof(T));
    return ptr;
  }

  void print(FILE* pFile)
  {
    assert(filled);
    for (int i = 0; i < nPoints; ++i) {
      float buf[] = {
        (float)ss[i], (float)pX[i], (float)pY[i], (float)pZ[i], (float)fX[i],
        (float)fY[i], (float)fZ[i], (float)vY[i], (float)vY[i], (float)vZ[i],
        (float)vxDef[i], (float)vyDef[i], (float)vzDef[i],
        (float)surface[i]->dchidx, (float)surface[i]->dchidy,
        (float)surface[i]->dchidz
      };
      fwrite (buf, sizeof(float), 16, pFile);
    }
  }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_ObstacleBlock_h
