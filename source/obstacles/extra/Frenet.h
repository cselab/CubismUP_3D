//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Wim van Rees on 17/10/14.
//

#ifndef CubismUP_3D_Frenet_h
#define CubismUP_3D_Frenet_h

#include "../../Definitions.h"

#include <limits>

CubismUP_3D_NAMESPACE_BEGIN

struct Frenet2D
{
  static void solve(const int Nm, const Real * const rS,
    const Real * const curv, const Real * const curv_dt,
    Real * const rX, Real * const rY, Real * const vX, Real * const vY,
    Real * const norX, Real * const norY, Real * const vNorX, Real * const vNorY)
    {
        // initial conditions
        rX[0] = 0.0;
        rY[0] = 0.0;
        norX[0] = 0.0;
        norY[0] = 1.0;
        Real ksiX = 1.0;
        Real ksiY = 0.0;

        // velocity variables
        vX[0] = 0.0;
        vY[0] = 0.0;
        vNorX[0] = 0.0;
        vNorY[0] = 0.0;
        Real vKsiX = 0.0;
        Real vKsiY = 0.0;

        for(int i=1; i<Nm; i++)
        {
            // compute derivatives positions
            const Real dksiX = curv[i-1]*norX[i-1];
            const Real dksiY = curv[i-1]*norY[i-1];
            const Real dnuX = -curv[i-1]*ksiX;
            const Real dnuY = -curv[i-1]*ksiY;

            // compute derivatives velocity
            const Real dvKsiX = curv_dt[i-1]*norX[i-1] + curv[i-1]*vNorX[i-1];
            const Real dvKsiY = curv_dt[i-1]*norY[i-1] + curv[i-1]*vNorY[i-1];
            const Real dvNuX = -curv_dt[i-1]*ksiX - curv[i-1]*vKsiX;
            const Real dvNuY = -curv_dt[i-1]*ksiY - curv[i-1]*vKsiY;

            // compute current ds
            const Real ds = rS[i] - rS[i-1];

            // update
            rX[i] = rX[i-1] + ds*ksiX;
            rY[i] = rY[i-1] + ds*ksiY;
            norX[i] = norX[i-1] + ds*dnuX;
            norY[i] = norY[i-1] + ds*dnuY;
            ksiX += ds * dksiX;
            ksiY += ds * dksiY;

            // update velocities
            vX[i] = vX[i-1] + ds*vKsiX;
            vY[i] = vY[i-1] + ds*vKsiY;
            vNorX[i] = vNorX[i-1] + ds*dvNuX;
            vNorY[i] = vNorY[i-1] + ds*dvNuY;
            vKsiX += ds * dvKsiX;
            vKsiY += ds * dvKsiY;

            // normalize unit vectors
            const Real d1 = ksiX*ksiX + ksiY*ksiY;
            const Real d2 = norX[i]*norX[i] + norY[i]*norY[i];
            if(d1>std::numeric_limits<Real>::epsilon())
            {
                const Real normfac = 1.0/std::sqrt(d1);
                ksiX*=normfac;
                ksiY*=normfac;
            }
            if(d2>std::numeric_limits<Real>::epsilon())
            {
                const Real normfac = 1.0/std::sqrt(d2);
                norX[i]*=normfac;
                norY[i]*=normfac;
            }
        }
    }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Frenet_h
