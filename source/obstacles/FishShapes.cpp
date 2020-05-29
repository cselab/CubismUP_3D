//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#include "FishShapes.h"

#include <gsl/gsl_bspline.h>
#include <gsl/gsl_statistics.h>
#include <iostream>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

void MidlineShapes::integrateBSpline(const double*const xc,
  const double*const yc, const int n, const double length,
  Real*const rS, Real*const res, const int Nm)
{
  double len = 0;
  for (int i=0; i<n-1; i++) {
    len += std::sqrt(std::pow(xc[i]-xc[i+1],2) +
                     std::pow(yc[i]-yc[i+1],2));
  }
  gsl_bspline_workspace *bw;
  gsl_vector *B;
  // allocate a cubic bspline workspace (k = 4)
  bw = gsl_bspline_alloc(4, n-2);
  B = gsl_vector_alloc(n);
  gsl_bspline_knots_uniform(0.0, len, bw);

  double ti = 0;
  for(int i=0; i<Nm; ++i) {
    res[i] = 0;
    if (rS[i]>0 and rS[i]<length) {
      const double dtt = (rS[i]-rS[i-1])/1e3;
      while (true) {
        double xi = 0;
        gsl_bspline_eval(ti, B, bw);
        for (int j=0; j<n; j++) xi += xc[j]*gsl_vector_get(B, j);
        if (xi >= rS[i]) break;
        if(ti + dtt > len)  break;
        else ti += dtt;
      }

      for (int j=0; j<n; j++) res[i] += yc[j]*gsl_vector_get(B, j);
    }
  }
  gsl_bspline_free(bw);
  gsl_vector_free(B);
}

void MidlineShapes::naca_width(const double t_ratio, const double L,
  Real*const rS, Real*const res, const int Nm)
{
  const Real a =  0.2969;
  const Real b = -0.1260;
  const Real c = -0.3516;
  const Real d =  0.2843;
  const Real e = -0.1015;
  const Real t = t_ratio*L;

  for(int i=0; i<Nm; ++i)
  {
    if ( rS[i]<=0 or rS[i]>=L ) res[i] = 0;
    else {
      const Real p = rS[i]/L;
      res[i] = 5*t* (a*std::sqrt(p) +b*p +c*p*p +d*p*p*p + e*p*p*p*p);
      /*
      if(s>0.99*L){ // Go linear, otherwise trailing edge is not closed - NACA analytical's fault
        const Real temp = 0.99;
        const Real y1 = 5*t* (a*std::sqrt(temp) +b*temp +c*temp*temp +d*temp*temp*temp + e*temp*temp*temp*temp);
        const Real dydx = (0-y1)/(L-0.99*L);
        return y1 + dydx * (s - 0.99*L);
      }else{ // NACA analytical
        return 5*t* (a*std::sqrt(p) +b*p +c*p*p +d*p*p*p + e*p*p*p*p);
      }
      */
    }
  }
}

void MidlineShapes::stefan_width(const double L, Real*const rS, Real*const res, const int Nm)
{
  const double sb = .04*L;
  const double st = .95*L;
  const double wt = .01*L;
  const double wh = .04*L;

  for(int i=0; i<Nm; ++i)
  {
    if(rS[i]<=0 or rS[i]>=L) res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] = (s<sb ? std::sqrt(2.0*wh*s - s*s) :
               (s<st ? wh -(wh-wt)*std::pow((s-sb)/(st-sb),2) :
               (       wt * (L-s)/(L-st))));
    }
  }
}

void MidlineShapes::stefan_height(const double L, Real*const rS, Real*const res, const int Nm)
{
  const double a=0.51*L;
  const double b=0.08*L;

  for(int i=0; i<Nm; ++i)
  {
    if(rS[i]<=0 or rS[i]>=L) res[i] = 0;
    else {
      const Real s = rS[i];
      res[i] = b*std::sqrt(1 - std::pow((s-a)/a,2));
    }
  }
}

void MidlineShapes::danio_width(const double L, Real*const rS, Real*const res, const int Nm)
{
	const int nBreaksW = 11;
	const double breaksW[nBreaksW] = {0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0};
	const double coeffsW[nBreaksW-1][4] = {
		{0.0015713,       2.6439,            0,       -15410},
		{ 0.012865,       1.4882,      -231.15,        15598},
		{0.016476,      0.34647,       2.8156,      -39.328},
		{0.032323,      0.38294,      -1.9038,       0.7411},
		{0.046803,      0.19812,      -1.7926,       5.4876},
		{0.054176,    0.0042136,     -0.14638,     0.077447},
		{0.049783,    -0.045043,    -0.099907,     -0.12599},
		{ 0.03577,     -0.10012,      -0.1755,      0.62019},
		{0.013687,      -0.0959,      0.19662,      0.82341},
		{0.0065049,     0.018665,      0.56715,       -3.781}
	};

	for(int i=0; i<Nm; ++i)
	{
		if ( rS[i]<=0 or rS[i]>=L ) res[i] = 0;
		else {

			const double sNormalized = rS[i]/L;

			// Find current segment
			int currentSegW = 1;
			while(sNormalized>=breaksW[currentSegW]) currentSegW++;
			currentSegW--; // Shift back to the correct segment
			//if(rS[i]==L) currentSegW = nBreaksW-2; Not necessary - excluded by the if conditional


			// Declare pointer for easy access
			const double *paramsW = coeffsW[currentSegW];
			// Reconstruct cubic
			const double xxW = sNormalized - breaksW[currentSegW];
			res[i] = L*(paramsW[0] + paramsW[1]*xxW + paramsW[2]*pow(xxW,2) + paramsW[3]*pow(xxW,3));
		}
	}
}

void MidlineShapes::danio_height(const double L, Real*const rS, Real*const res, const int Nm)
{
	const int nBreaksH = 15;
	const double breaksH[nBreaksH] = {0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.87, 0.9, 0.993, 0.996, 0.998, 1};
	const double coeffsH[nBreaksH-1][4] = {
		{0.0011746,        1.345,   2.2204e-14,      -578.62},
		{0.014046,       1.1715,      -17.359,        128.6},
		{0.041361,     0.40004,      -1.9268 ,      9.7029},
		{0.057759,      0.28013,     -0.47141,     -0.08102},
		{0.094281,     0.081843,     -0.52002,     -0.76511},
		{0.083728,     -0.21798,     -0.97909,       3.9699},
		{0.032727,     -0.13323,       1.4028,       2.5693},
		{0.036002,      0.22441,       2.1736,      -13.194},
		{0.051007,      0.34282,      0.19446,       16.642},
		{0.058075,      0.37057,        1.193,      -17.944},
		{0.069781,       0.3937,     -0.42196,      -29.388},
		{0.079107,     -0.44731,      -8.6211,  -1.8283e+05},
		{0.072751,      -5.4355,      -1654.1,  -2.9121e+05},
		{0.052934,      -15.546,      -3401.4,   5.6689e+05}
	};

	for(int i=0; i<Nm; ++i)
	{
		if ( rS[i]<=0 or rS[i]>=L ) res[i] = 0;
		else {

			const double sNormalized = rS[i]/L;

			// Find current segment
			int currentSegH = 1;
			while(sNormalized>=breaksH[currentSegH]) currentSegH++;
			currentSegH--; // Shift back to the correct segment
			//if(rS[i]==L) currentSegH = nBreaksH-2; Not necessary - excluded by the if conditional

			// Declare pointer for easy access
			const double *paramsH = coeffsH[currentSegH];
			// Reconstruct cubic
			const double xxH = sNormalized - breaksH[currentSegH];
			res[i] = L*(paramsH[0] + paramsH[1]*xxH + paramsH[2]*pow(xxH,2) + paramsH[3]*pow(xxH,3));
		}
	}
}

void MidlineShapes::computeWidthsHeights(
    const std::string &heightName,
    const std::string &widthName,
    const double L,
    Real* const rS,
    Real* const height,
    Real* const width,
    const int nM,
    const int mpirank)
{
  using std::cout;
  using std::endl;
  if(!mpirank) {
    printf("height = %s, width=%s\n", heightName.c_str(), widthName.c_str());
    fflush(NULL);
  }

  {
    if ( heightName.compare("largefin") == 0 ) {
      if(!mpirank)
        cout<<"Building object's height according to 'largefin' profile."<<endl;
      double xh[8] = {0, 0, .2*L, .4*L, .6*L, .8*L, L, L};
      double yh[8] = {0, .055*L, .18*L, .2*L, .064*L, .002*L, .325*L, 0};
      // TODO read second to last number from factory
      integrateBSpline(xh, yh, 8, L, rS, height, nM);
    } else
    if ( heightName.compare("tunaclone") == 0 ) {
      if(!mpirank)
        cout<<"Building object's height according to 'tunaclone' profile."<<endl;
      double xh[9] = {0, 0, 0.2*L, .4*L, .6*L, .9*L, .96*L, L, L};
      double yh[9] = {0, .05*L, .14*L, .15*L, .11*L, 0, .1*L, .2*L, 0};
      integrateBSpline(xh, yh, 9, L, rS, height, nM);
    } else
    if ( heightName.compare(0, 4, "naca") == 0 ) {
      double t_naca = std::stoi( heightName.substr(5), nullptr, 10 ) * 0.01;
      if(!mpirank)
        cout<<"Building object's height according to naca profile with adim. thickness param set to "<<t_naca<<" ."<<endl;
      naca_width(t_naca, L, rS, height, nM);
    } else
    if ( heightName.compare("danio") == 0 ) {
      if(!mpirank)
        cout<<"Building object's height according to Danio (zebrafish) profile from Maertens2017 (JFM)"<<endl;
      danio_height(L, rS, height, nM);
    } else
    if ( heightName.compare("stefan") == 0 ) {
      if(!mpirank)
        cout<<"Building object's height according to Stefan profile"<<endl;
      stefan_height(L, rS, height, nM);
    } else {
      if(!mpirank)
        cout<<"Building object's height according to baseline profile."<<endl;
      double xh[8] = {0, 0, .2*L, .4*L, .6*L, .8*L, L, L};
      double yh[8] = {0, .055*L, .068*L, .076*L, .064*L, .0072*L, .11*L, 0};
      integrateBSpline(xh, yh, 8, L, rS, height, nM);
    }
  }



  {
    if ( widthName.compare("fatter") == 0 ) {
      if(!mpirank)
        cout<<"Building object's width according to 'fatter' profile."<<endl;
      double xw[6] = {0, 0, L/3., 2*L/3., L, L};
      double yw[6] = {0, 8.9e-2*L, 7.0e-2*L, 3.0e-2*L, 2.0e-2*L, 0};
      integrateBSpline(xw, yw, 6, L, rS, width, nM);
    } else
    if ( widthName.compare(0, 4, "naca") == 0 ) {
      double t_naca = std::stoi( widthName.substr(5), nullptr, 10 ) * 0.01;
      if(!mpirank)
        cout<<"Building object's width according to naca profile with adim. thickness param set to "<<t_naca<<" ."<<endl;
      naca_width(t_naca, L, rS, width, nM);
    } else
    if ( widthName.compare("danio") == 0 ) {
      if(!mpirank)
        cout<<"Building object's width according to Danio (zebrafish) profile from Maertens2017 (JFM)"<<endl;
      danio_width(L, rS, width, nM);
    } else
    if ( widthName.compare("stefan") == 0 ) {
      if(!mpirank)
        cout<<"Building object's width according to Stefan profile"<<endl;
      stefan_width(L, rS, width, nM);
    } else {
      if(!mpirank)
        cout<<"Building object's width according to baseline profile."<<endl;
      double xw[6] = {0, 0, L/3., 2*L/3., L, L};
      double yw[6] = {0, 8.9e-2*L, 1.7e-2*L, 1.6e-2*L, 1.3e-2*L, 0};
      integrateBSpline(xw, yw, 6, L, rS, width, nM);
    }
  }

  if(!mpirank) {
    FILE * heightWidth;
    heightWidth = fopen("widthHeight.dat","w");
    for(int i=0; i<nM; ++i)
      fprintf(heightWidth,"%.8e \t %.8e \t %.8e \n", rS[i], width[i], height[i]);
    fclose(heightWidth);
  }
}

CubismUP_3D_NAMESPACE_END
