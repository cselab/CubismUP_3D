//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Mattia Gazzola on 4/1/11.
//

#ifndef CubismUP_3D_Interpolation1D_h
#define CubismUP_3D_Interpolation1D_h

#include "../../Base.h"

#include <cstdio>

CubismUP_3D_NAMESPACE_BEGIN

class Interpolation1D
{
 public:
  template <typename T>
  static void naturalCubicSpline(const double * x, const double * y, const unsigned int n, const T * xx, T * yy, const unsigned int nn)
  {
      return naturalCubicSpline(x,y,n,xx,yy,nn,0);
  }
  template <typename T>
  static void naturalCubicSpline(const double * x, const double * y, const unsigned int n, const T * xx, T * yy, const unsigned int nn, const Real offset)
  {
    double* y2  = new double[n];
    double* u  = new double[n-1];

    y2[0] = u[0] = 0.0;
    for(unsigned int i=1; i<n-1; i++)
    {
      const double sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
      const double p=sig*y2[i-1]+2.0;
      y2[i]=(sig-1.0)/p;
      u[i]=(y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]);
      u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
    }
    y2[n-1]=0;

    for(unsigned int k=n-2; k>0; k--) y2[k]=y2[k]*y2[k+1]+u[k];

    for(unsigned int j=0; j<nn; j++)
    {
      unsigned int klo = 0;
      unsigned int khi = n-1;
      unsigned int k = 0;
      while(khi-klo>1) {
        k=(khi+klo)>>1;
        if( x[k]>(xx[j]+offset)) khi=k;
        else klo=k;
      }

      const double h = x[khi]-x[klo];
      if(abs(h)<2.2e-16) {
        printf("Interpolation points must be distinct!"); fflush(0); abort();
      }
      const double a = (x[khi]-(xx[j]+offset))/h;
      const double b = ((xx[j]+offset)-x[klo])/h;
      yy[j] = a*y[klo]+b*y[khi]+((a*a*a-a)*y2[klo]+(b*b*b-b)*y2[khi])*(h*h)/6;
    }

    delete [] y2;
    delete [] u;
  }

  template <typename T>
  static void cubicInterpolation(const double x0, const double x1, const double x, const double y0, const double y1, const double dy0, const double dy1, T & y, T & dy)
  {
    const double xrel = (x-x0);
    const double deltax = (x1-x0);

    const double a = (dy0+dy1)/(deltax*deltax) - 2*(y1-y0)/(deltax*deltax*deltax);
    const double b = (-2*dy0-dy1)/deltax + 3*(y1-y0)/(deltax*deltax);
    const double c = dy0;
    const double d = y0;

    y = a*xrel*xrel*xrel + b*xrel*xrel + c*xrel + d;
    dy = 3*a*xrel*xrel + 2*b*xrel + c;
  }

  template <typename T>
  static void cubicInterpolation(const double x0, const double x1, const double x, const double y0, const double y1, T & y, T & dy)
  {
      return cubicInterpolation(x0,x1,x,y0,y1,0.0,0.0,y,dy); // zero slope at end points
  }

};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Interpolation1D_h
