//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Pantelis Vlachas.
//

#include "obstacles/IF3D_ElasticFishOperator.h"
#include "obstacles/IF3D_FishLibrary.h"
#include "operators/GenericOperator.h"

#include "Cubism/ArgumentParser.h"
#include "Cubism/HDF5Dumper_MPI.h"

#include <cmath>

class ElasticMidlineData : public FishMidlineData
{
  Real * const theta;
  Real * const thetaOld;
  Real * const cosTheta;
  Real * const dTheta;
  Real * const linkLength;
  Real * const linkWidth;
  Real * const linkMass;
  Real * const linkInertia;
  Real * const elasticityCoefficients;
  Real * const dampingCoefficients;
  Real * const forcesElasticPlusDampingX;
  Real * const forcesElasticPlusDampingY;
  Real * const forcesTotalX;
  Real * const forcesTotalY;


  Real * const vX_plus_half;
  Real * const vY_plus_half;
  Real * const vX_minus_half;
  Real * const vY_minus_half;

  double sim_dt = -1000;
  double sim_time = -1000;
  bool started_deforming = false;
  double time_start_deforming = 0.02;// 1e-3;
  double epsilon = 1e-17;


  double rattle_accuracy = 1e-10;
  int rattle_max_iter = 1e7;

 public:
  ElasticMidlineData(const double L, const double _h, double zExtent, double t_ratio, double HoverL=1) :
    FishMidlineData(L, 1, 0, _h),
    theta(_alloc(Nm-2)),
    thetaOld(_alloc(Nm-2)),
    cosTheta(_alloc(Nm-2)),
    dTheta(_alloc(Nm-2)),
    linkLength(_alloc(Nm-1)),
    linkWidth(_alloc(Nm-1)),
    linkMass(_alloc(Nm-1)),
    linkInertia(_alloc(Nm-1)),
    elasticityCoefficients(_alloc(Nm-2)),
    dampingCoefficients(_alloc(Nm-2)),
    forcesElasticPlusDampingX(_alloc(Nm)),
    forcesElasticPlusDampingY(_alloc(Nm)),
    forcesTotalX(_alloc(Nm)),
    forcesTotalY(_alloc(Nm)),
    vX_plus_half(_alloc(Nm)),
    vY_plus_half(_alloc(Nm)),
    vX_minus_half(_alloc(Nm)),
    vY_minus_half(_alloc(Nm))
  {
    for(int i=0;i<Nm;++i) height[i] = length*HoverL/2;
    MidlineShapes::naca_width(t_ratio, length, rS, width, Nm);

    computeMidline(0.0, 1.0);

    #if 1
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD,&rank);
      if (rank!=0) return;
      FILE * f = fopen("fish_profile","w");
      for (int i=0; i<Nm; ++i) fprintf(f, "%g %g %g %g %g\n",
      rX[i],rY[i],rS[i],width[i],height[i]);
      fflush(f); fclose(f);
    #endif
  }


  bool timeIsZero(const double time_)
  {
    return time_ < epsilon;
  }

  double r2()
  {
    return (double)rand() / (double)RAND_MAX ;
  }

  double dotProduct(const double x1, const double y1, const double x2, const double y2)
  {
    return x1*x2 + y1*y2;
  }

  void computeAngleDerivatives(const Real * const theta, const Real * const thetaOld, Real * dTheta)
  {
    for(int i=0; i<Nm-2; ++i){
      dTheta[i] = (theta[i] - thetaOld[i]) / sim_dt;
    }
  }

  void computeAngles(const Real * const rX, const Real * const rY, Real * cosTheta, Real * theta)
  {
    for(int i=0; i<Nm-2; ++i){
      double r1x, r1y, r2x, r2y, r3x, r3y, r23x, r23y, r21x, r21y, temp, crossproduct;
      r1x = rX[i];
      r2x = rX[i+1];
      r3x = rX[i+2];
      r1y = rY[i];
      r2y = rY[i+1];
      r3y = rY[i+2];
      r23x = r3x - r2x;
      r21x = r1x - r2x;
      r23y = r3y - r2y;
      r21y = r1y - r2y;
      temp = dotProduct(r23x,r23y,r21x,r21y) / std::sqrt(dotProduct(r23x,r23y,r23x,r23y)*dotProduct(r21x,r21y,r21x,r21y));
      if(temp > 1.0){
        temp = 1.0;
      }else if(temp < -1.0){
        temp = -1.0;
      }
      cosTheta[i] = temp;

      crossproduct = r21x*r23y - r21y*r23x;

      // Mapping the angles to the range [0, 2*pi]
      if(crossproduct > 0.0){
        theta[i] = 2*M_PI - std::acos(temp);
      }else{
        theta[i] = std::acos(temp);
      }

      // Assert angles bigger than zero
      assert(theta[i] >= 0.0);
      // Assert valid range for angles in [pi/2, 3*pi/2]
      assert(theta[i] >= M_PI/2.0);
      assert(theta[i] <= 3*M_PI/2.0);

      // printf("theta = %le \n", theta[i]);
    }
  }

  void computeLinkLength(const Real * const rS, Real * linkLength)
  {
    printf("MIDLINE: Computing link length.\n");
    for(int i=0; i<Nm-1; ++i){
      linkLength[i] = rS[i+1]-rS[i];
      assert(linkLength[i]>epsilon);
    }
    printf("MIDLINE: Link length computed!\n");

    // // DOUBLE CHECKING
    // for(int i=0; i<Nm-1; ++i) {
    //   printf("linkLength = %le \n", linkLength[i]);
    // }
  }

  void computeLinkWidth(const Real * const width, Real * linkWidth)
  {
    printf("MIDLINE: Computing link width.\n");
    // linkWidth is the mean width of the two nodes. (Recall width[i] is the half width of the node)
    for(int i=0;i<Nm-1;++i){
      linkWidth[i] = width[i]+width[i+1];
      assert(linkWidth[i]>epsilon);
    }
    printf("MIDLINE: Link width computed!\n");

    // // DOUBLE CHECKING
    // for(int i=0; i<Nm-1; ++i) {
    //   printf("linkWidth = %le \n", linkWidth[i]);
    // }
  }



  void computeElasticLinkForces(const Real * const rX, const Real * const rY, const Real * const elasticityCoefficients, const Real * const dampingCoefficients, const Real * const theta, const Real * const dTheta, const Real * const cosTheta, Real * forcesElasticPlusDampingX, Real * forcesElasticPlusDampingY)
  {
    printf("MIDLINE: Computing elastic link forces.\n");

    // Initialize all forces to zero
    for(int i=0; i<Nm; ++i){
      forcesElasticPlusDampingX[i] = 0.0;
      forcesElasticPlusDampingY[i] = 0.0;
    }


    for(int i=0; i<Nm-2; ++i){
      double r1x = rX[i];
      double r1y = rY[i];

      double r2x = rX[i+1];
      double r2y = rY[i+1];

      double r3x = rX[i+2];
      double r3y = rY[i+2];

      double r23x = r3x - r2x;
      double r23y = r3y - r2y;

      double r21x = r1x - r2x;
      double r21y = r1y - r2y;

      double r21_norm = std::sqrt(dotProduct(r21x,r21y,r21x,r21y));
      double r23_norm = std::sqrt(dotProduct(r23x,r23y,r23x,r23y));

      double sintheta_abs = sqrt(1-cosTheta[i]*cosTheta[i]);
      if (sintheta_abs < epsilon){
        sintheta_abs = epsilon;
      }

      double dU_dcostheta = - elasticityCoefficients[i] * (theta[i] - M_PI)/sintheta_abs;
      double dP_dtheta_dot_scaled = -dampingCoefficients[i] * dTheta[i] / sintheta_abs;

      double a11 = cosTheta[i]/(r21_norm*r21_norm);
      double a22 = cosTheta[i]/(r23_norm*r23_norm);
      double a12 = -1.0/(r21_norm*r23_norm);

      double F21x = (dU_dcostheta + dP_dtheta_dot_scaled)* (a11*r21x + a12*r23x);
      double F21y = (dU_dcostheta + dP_dtheta_dot_scaled) * (a11*r21y + a12*r23y);

      double F23x = (dU_dcostheta + dP_dtheta_dot_scaled) * (a12*r21x + a22*r23x);
      double F23y = (dU_dcostheta + dP_dtheta_dot_scaled) * (a12*r21y + a22*r23y);

      forcesElasticPlusDampingX[i] += F21x;
      forcesElasticPlusDampingX[i+1] += (-F21x - F23x);
      forcesElasticPlusDampingX[i+2] += F23x;

      forcesElasticPlusDampingY[i] += F21y;
      forcesElasticPlusDampingY[i+1] += (-F21y - F23y);
      forcesElasticPlusDampingY[i+2] += F23y;
    }

    // Checking that the elasticity forces have a valid range
    for(int i=0; i<Nm; ++i){
      assert(forcesElasticPlusDampingX[i]>=0.0);
      assert(forcesElasticPlusDampingX[i]<=1e2);
      assert(forcesElasticPlusDampingY[i]>=0.0);
      assert(forcesElasticPlusDampingY[i]<=1e2);
    }
    printf("MIDLINE: Elastic link forces computed!\n");
  }

  void setEqualVectors(const int dim, const Real * const x1, Real * x2)
  {
    for(int i=0; i<dim; ++i){
      x2[i] = x1[i];
    }
  }

  void computeTotalForces(const Real * const f1X, const Real * const f1Y, const Real * const f2X, const Real * const f2Y, Real * totalForceX, Real * totalForceY )
  {
    printf("MIDLINE: Computing total forces.\n");
    // Setting the forces initially to zero
    for(int i=0; i<Nm; ++i)
    {
      totalForceX[i] = 0.0;
      totalForceY[i] = 0.0;
    }
    // Computing the total forces
    for(int i=0; i<Nm; ++i)
    {
      totalForceX[i] = f1X[i] + f2X[i];
      totalForceY[i] = f1Y[i] + f2Y[i];
    }

    // Check for reasonable total forces
    for(int i=0; i<Nm; ++i)
    {
      assert(totalForceX[i] >= 0.0);
      assert(totalForceX[i] <= 1e2);
      assert(totalForceY[i] >= 0.0);
      assert(totalForceY[i] <= 1e2);
    }

    printf("MIDLINE: Total forces computed!\n");
  }

  void rattleComputeVelocityConstraintResiduals(const Real * const rX, const Real * const rY, const Real * const vX, const Real * const vY, const Real * const linkLength, double & total_residual, double & max_residual, int & max_residual_idx)
  {
    // Calculate all residuals
    // Number of constraints is equal with the number of links! (Nm-1)
    total_residual = 0.0;
    max_residual = -100.0;
    max_residual_idx = 0;
    for(int c=0; c<Nm-1; ++c)
    {
      double rX1 = rX[c];
      double rX2 = rX[c+1];
      double vX1 = vX[c];
      double vX2 = vX[c+1];

      double rY1 = rY[c];
      double rY2 = rY[c+1];
      double vY1 = vY[c];
      double vY2 = vY[c+1];

      double rX12 = rX1 - rX2;
      double rY12 = rY1 - rY2;
      double vX12 = vX1 - vX2;
      double vY12 = vY1 - vY2;

      double sigmadot = dotProduct(vX12, vY12, rX12, rY12);
      double residual = std::fabs(sigmadot) / Nm;

      if (residual > max_residual)
      {
        max_residual = residual;
        max_residual_idx = c;
        total_residual = total_residual + residual;
      }
    }
  }

  void rattleComputePositionConstraintResiduals(const Real * const rX, const Real * const rY, const Real * const vX, const Real * const vY, const Real * const linkLength, double & total_residual, double & max_residual, int & max_residual_idx)
  {
    // Calculate all residuals
    // In this case dr_ = dr_onehalf
    // Number of constraints is equal with the number of links! (Nm-1)
    total_residual = 0.0;
    max_residual = -100.0;
    max_residual_idx = 0;
    for(int c=0; c<Nm-1; ++c)
    {
      double rX1 = rX[c];
      double rX2 = rX[c+1];
      double vX1 = vX[c];
      double vX2 = vX[c+1];

      double rY1 = rY[c];
      double rY2 = rY[c+1];
      double vY1 = vY[c];
      double vY2 = vY[c+1];

      double ssX = rX1 + sim_dt*vX1 - rX2 - sim_dt*vX2;
      double ssY = rY1 + sim_dt*vY1 - rY2 - sim_dt*vY2;

      double ss_norm = std::sqrt(dotProduct(ssX, ssY, ssX, ssY));
      double residual = std::fabs(ss_norm - linkLength[c]) / Nm;

      // printf("residual = %le \n", residual);
      // printf("rX1 = %le \n", rX1);
      // printf("rX2 = %le \n", rX2);
      // printf("vX1 = %le \n", vX1);
      // printf("vX2 = %le \n", vX2);
      // printf("ss_norm = %le \n", ss_norm);
      // printf("linkLength = %le \n", linkLength[c]);

      if (residual > max_residual)
      {
        max_residual = residual;
        max_residual_idx = c;
        total_residual = total_residual + residual;
      }
    }
  }

  void rattleVelocityUpdateZero(const Real * const rX_0, const Real * const rY_0, const Real * const vX_0, const Real * const vY_0, const Real * const forceX, const Real * const forceY, const Real * const mass, const Real * const linkLength, Real * vX_half, Real * vY_half)
  {
    for(int i=0; i<Nm; ++i){
      vX_half[i] = vX_0[i] + forceX[i]*sim_dt/2.0/mass[i];
      vY_half[i] = vY_0[i] + forceY[i]*sim_dt/2.0/mass[i];
    }

    int iter = 1;
    double max_residual, total_residual;
    int max_residual_idx;
    rattleComputePositionConstraintResiduals(rX_0,rY_0,vX_half,vY_half, linkLength,total_residual, max_residual, max_residual_idx);

    while(max_residual > rattle_accuracy and iter < rattle_max_iter){
      int i = max_residual_idx;
      int j = i + 1;

      double mi = mass[i];
      double mj = mass[j];

      double r_ij_x = rX_0[i] - rX_0[j];
      double r_ij_y = rY_0[i] - rY_0[j];

      double v_half_ij_X = vX_half[i] - vX_half[j];
      double v_half_ij_Y = vY_half[i] - vY_half[j];

      double ssX = r_ij_x + sim_dt * v_half_ij_X;
      double ssY = r_ij_y + sim_dt * v_half_ij_Y;

      double ss = dotProduct(ssX, ssY, ssX, ssY);
      ss = ss - linkLength[i]*linkLength[i];

      double delta = ss / ( 2.0 * sim_dt * dotProduct(ssX, ssY, r_ij_x, r_ij_y) * (1.0/mi + 1.0/mj));

      vX_half[i] = vX_half[i] - delta/mi*r_ij_x;
      vY_half[i] = vY_half[i] - delta/mi*r_ij_y;

      vX_half[j] = vX_half[j] + delta/mj*r_ij_x;
      vY_half[j] = vY_half[j] + delta/mj*r_ij_y;

      iter = iter + 1;
      rattleComputePositionConstraintResiduals(rX_0,rY_0,vX_half,vY_half, linkLength, total_residual, max_residual, max_residual_idx);

      if(iter % int(1e5) == 0.0){
          printf("RATTLE VEL UPDATE ZERO: iter= %d, max_residual_idx = %d, max_residual = %le, total_residual = %le \n", iter, max_residual_idx, max_residual, total_residual);
      }
    }
    printf("#################### RATTLE VEL UPDATE ZERO  ####################\n");
    printf("# TIME = %le, dt= %le, Method ended with error = %le after = %d iterations #\n", sim_time, sim_dt, max_residual, iter);
    printf("#################################################################\n");


    for(int i=0; i<Nm; ++i){
      assert(vX_half[i] >=0 );
      assert(vX_half[i] <=1e2 );
      assert(vY_half[i] >=0 );
      assert(vY_half[i] <=1e2 );
    }
  }

  void rattleVelocityUpdateHalf(const Real * const rX_n, const Real * const rY_n, const Real * const vX_n_minus_half, const Real * const vY_n_minus_half, const Real * const forceX, const Real * const forceY, const Real * const mass, const Real * const linkLength, Real * vX_n_plus_half, Real * vY_n_plus_half)
  {

    for(int i=0; i<Nm; ++i){
      vX_n_plus_half[i] = vX_n_minus_half[i] + forceX[i]*sim_dt/mass[i];
      vY_n_plus_half[i] = vY_n_minus_half[i] + forceY[i]*sim_dt/mass[i];
    }

    int iter = 1;
    double max_residual, total_residual;
    int max_residual_idx;
    rattleComputePositionConstraintResiduals(rX_n,rY_n,vX_n_plus_half,vY_n_plus_half, linkLength,total_residual, max_residual, max_residual_idx);

    while(max_residual > rattle_accuracy and iter < rattle_max_iter){
      int i = max_residual_idx;
      int j = i + 1;

      double mi = mass[i];
      double mj = mass[j];

      double r_ij_x = rX_n[i] - rX_n[j];
      double r_ij_y = rY_n[i] - rY_n[j];

      double vX_n_plus_half_ij = vX_n_plus_half[i] - vX_n_plus_half[j];
      double vY_n_plus_half_ij = vY_n_plus_half[i] - vY_n_plus_half[j];

      double ssX = r_ij_x + sim_dt * vX_n_plus_half_ij;
      double ssY = r_ij_y + sim_dt * vY_n_plus_half_ij;

      double ss = dotProduct(ssX, ssY, ssX, ssY);
      ss = ss - linkLength[i]*linkLength[i];

      double delta = ss / ( 2.0 * sim_dt * dotProduct(ssX, ssY, r_ij_x, r_ij_y) * (1/mi + 1/mj));

      vX_n_plus_half[i] = vX_n_plus_half[i] - delta/mi*r_ij_x;
      vY_n_plus_half[i] = vY_n_plus_half[i] - delta/mi*r_ij_y;

      vX_n_plus_half[j] = vX_n_plus_half[j] + delta/mj*r_ij_x;
      vY_n_plus_half[j] = vY_n_plus_half[j] + delta/mj*r_ij_y;

      iter = iter + 1;
      rattleComputePositionConstraintResiduals(rX_n,rY_n,vX_n_plus_half,vY_n_plus_half, linkLength,total_residual, max_residual, max_residual_idx);

      if(iter % int(1e5) == 0.0){
          printf("RATTLE VEL UPDATE HALF: iter= %d, max_residual_idx = %d, max_residual = %le, total_residual = %le \n", iter, max_residual_idx, max_residual, total_residual);

      }
    }
    printf("#################### RATTLE VEL UPDATE HALF  ####################\n");
    printf("# TIME = %le, dt= %le, Method ended with error = %le after = %d iterations #\n", sim_time, sim_dt, max_residual, iter);
    printf("#####################################################################\n");
  }

  void rattleVelocityUpdate(const Real * const rX_n, const Real * const rY_n,const Real * const vX_n_minus_half, const Real * const vY_n_minus_half, const Real * const vX_n_plus_half, const Real * const vY_n_plus_half, const Real * const mass, const Real * const linkLength, Real * vX_n, Real * vY_n)
  {
    for(int i=0; i<Nm; ++i){
      vX_n[i] = (vX_n_minus_half[i] + vX_n_plus_half[i])/2.0;
      vY_n[i] = (vY_n_minus_half[i] + vY_n_plus_half[i])/2.0;
    }
    int iter = 1;
    double max_residual, total_residual;
    int max_residual_idx;

    rattleComputeVelocityConstraintResiduals(rX_n,rY_n,vX_n,vY_n,linkLength,total_residual, max_residual, max_residual_idx);
    while(max_residual > rattle_accuracy and iter < rattle_max_iter){
      int i = max_residual_idx;
      int j = i + 1;

      double mi = mass[i];
      double mj = mass[j];

      double r_ij_x = rX_n[i] - rX_n[j];
      double r_ij_y = rY_n[i] - rY_n[j];
      double v_ij_x = vX_n[i] - vX_n[j];
      double v_ij_y = vY_n[i] - vY_n[j];

      double kappa = dotProduct(v_ij_x, v_ij_y, r_ij_x, r_ij_y) / dotProduct(r_ij_x, r_ij_y, r_ij_x, r_ij_y) / (1.0/mi + 1.0/mj);

      vX_n[i] = vX_n[i] - kappa / mi * r_ij_x;
      vY_n[i] = vY_n[i] - kappa / mi * r_ij_x;

      vX_n[j] = vX_n[j] + kappa / mj * r_ij_x;
      vY_n[j] = vY_n[j] + kappa / mj * r_ij_x;

      rattleComputeVelocityConstraintResiduals(rX_n,rY_n,vX_n,vY_n,linkLength,total_residual, max_residual, max_residual_idx);

      iter = iter + 1;

      if(iter % int(1e5) == 0.0){
        printf("RATTLE VEL UPDATE: iter= %d, max_residual_idx = %d, max_residual = %le, total_residual = %le \n", iter, max_residual_idx, max_residual, total_residual);
      }
    }
    printf("#################### RATTLE VEL UPDATE  ####################\n");
    printf("# TIME = %le, dt= %le, Method ended with error = %le after = %d iterations #\n", sim_time, sim_dt, max_residual, iter);
    printf("#####################################################################\n");
  }


  void rattleIntegrationStep(const Real * const vX, const Real * const vY, Real * rX, Real * rY)
  {
    for(int i=0; i<Nm; ++i)
    {
      rX[i] = rX[i] + sim_dt*vX[i];
      rY[i] = rY[i] + sim_dt*vY[i];
      assert(rX[i]>=0.0);
      assert(rX[i]<=1.0);
      assert(rY[i]>=0.0);
      assert(rY[i]<=1.0);
    }
  }

  void computeLinkMass(Real * mass)
  {
    printf("MIDLINE: Computing link masses.\n");

    for(int i=0; i<Nm-1; ++i){
      mass[i] = linkWidth[i] * linkLength[i];
      assert(mass[i]>epsilon);
    }

    printf("MIDLINE: Link masses computed.\n");
  }

  void computeLinkInertia(Real * inertia)
  {
    printf("MIDLINE: Computing link Inertia.\n");

    for(int i=0; i<Nm-2; ++i){
      double inertia_left = linkMass[i]*(pow(linkLength[i]/2.0, 2.0)/3.0 + pow(linkWidth[i], 2.0)/12.0 );
      double inertia_right = linkMass[i+1]*(pow(linkLength[i+1]/2.0, 2.0)/3.0 + pow(linkWidth[i+1], 2.0)/12.0 );
      inertia[i] = inertia_left + inertia_right;
      assert(inertia[i]>epsilon);
    }
    printf("MIDLINE: Link Inertia computed.\n");
  }

  void computeElasticityCoefficients(Real * elasticityCoefficients)
  {
    printf("MIDLINE: Computing Link elasticity coefficients.\n");
    double E = 10e3;
    for(int i=0; i<Nm-2; ++i){
      elasticityCoefficients[i] = E*linkInertia[i];
      assert(elasticityCoefficients[i]>epsilon);
    }
    printf("MIDLINE: Link elasticity coefficients computed.\n");
  }

  void computeDampingCoefficients(Real * dampingCoefficients)
  {
    printf("MIDLINE: Computing Link damping coefficients.\n");
    for(int i=0;i<Nm-2;++i){
      double mean_mass = (linkMass[i] + linkMass[i+1])/2.0;
      dampingCoefficients[i] = sqrt(elasticityCoefficients[i]*mean_mass);
    }
    printf("MIDLINE: Link damping coefficients computed.\n");
  }

  void initializeThetaOld(){
    for(int i=0;i<Nm-2;++i){
      thetaOld[i] = theta[i];
      dTheta[i] = 0.0;
    }
  }

  void computeMidline(const double time, const double dt) override
  {
    sim_time = time;

    rX[0] = rY[0] = vX[0] = vY[0] = 0;
    for(int i=1; i<Nm; ++i) {
      rY[i] = vX[i] = vY[i] = 0;
      rX[i] = rX[i-1] + std::fabs(rS[i]-rS[i-1]);
    }
    _computeMidlineNormals();


/*
      const std::array<double ,6> curvature_points = {
          0, .15*length, .4*length, .65*length, .9*length, length
      };
      const std::array<double ,6> curvature_values = {
          0.82014/length, 1.46515/length, 2.57136/length,
          3.75425/length, 5.09147/length, 5.70449/length
      };
      curvScheduler.transition(time,0,1,curvature_values,curvature_values);
      // query the schedulers for current values
      curvScheduler.gimmeValues(time, curvature_points, Nm, rS, rC, vC);
      // construct the curvature
      for(int i=0; i<Nm; i++) {
        const double darg = 2.*M_PI;
        const double arg  = 2.*M_PI*(time -rS[i]/length) + M_PI*phaseShift;
        rK[i] =   rC[i]*std::sin(arg);
        vK[i] =   vC[i]*std::sin(arg) + rC[i]*std::cos(arg)*darg;
      }

      // solve frenet to compute midline parameters
      IF2D_Frenet2D::solve(Nm, rS, rK, vK, rX, rY, vX, vY, norX, norY, vNorX, vNorY);
      */
  }
};

IF3D_ElasticFishOperator::IF3D_ElasticFishOperator(SimulationData&s, ArgumentParser&p) : IF3D_FishOperator(s, p)
{
  const double thickness = p("-thickness").asDouble(0.12); // (NON DIMENSIONAL)
  bBlockRotation[0] = true;
  bBlockRotation[1] = true;
  myFish = new ElasticMidlineData(length, vInfo[0].h_gridpoint, ext_Z, thickness);
}

void IF3D_ElasticFishOperator::writeSDFOnBlocks(const mapBlock2Segs& segmentsPerBlock)
{
  #pragma omp parallel
  {
    PutNacaOnBlocks putfish(myFish, position, quaternion);

    #pragma omp for schedule(dynamic)
    for(size_t i=0; i<vInfo.size(); i++) {
      BlockInfo info = vInfo[i];
      const auto pos = segmentsPerBlock.find(info.blockID);
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;

      if(pos != segmentsPerBlock.end()) {
        for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
          b(ix,iy,iz).tmpU = 0.; //this will be accessed with plus equal

        assert(obstacleBlocks.find(info.blockID) != obstacleBlocks.end());
        ObstacleBlock*const block = obstacleBlocks.find(info.blockID)->second;
        putfish(info, b, block, pos->second);
      }
    }
  }

  #if 0
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < (int)vInfo.size(); ++i) {
      BlockInfo info = vInfo[i];
      const auto pos = obstacleBlocks.find(info.blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        //b(ix,iy,iz).chi = pos->second->chi[iz][iy][ix];//b(ix,iy,iz).tmpU;
        b(ix,iy,iz).u = b(ix,iy,iz).tmpU;
        b(ix,iy,iz).v = b(ix,iy,iz).tmpV;
        b(ix,iy,iz).w = b(ix,iy,iz).tmpW;
      }
    }
  }
  DumpHDF5_MPI<StreamerVelocityVector, DumpReal>(*grid, 0, 0, "SFD", "./");
  abort();
  #endif
}

void IF3D_ElasticFishOperator::computeForces(const int stepID, const double time, const double dt, const Real* Uinf, const double NU, const bool bDump)
{
  IF3D_ObstacleOperator::computeForces(stepID, time, dt, Uinf, NU, bDump);
  // This obstacle requires forces and torques on the midline. War plan:
  // 0) Fetch
  const int Nm = myFish->Nm;
  double * const fX = myFish->forceX;
  double * const fY = myFish->forceY;
  double * const tZ = myFish->torque;
  const Real*const pX = myFish->rX;
  const Real*const pY = myFish->rY;
  // 1) Reset
  std::fill(fX, fX+Nm, 0.0);
  std::fill(fY, fY+Nm, 0.0);
  std::fill(tZ, tZ+Nm, 0.0);
  // 2) Sum surface forces to the closest midline point using section marker
  for (const auto& pos : obstacleBlocks) {
    const ObstacleBlock* const o = pos.second;
    for(int i=0; i<o->nPoints; i++) {
      const int ss = o->ss[i];
      assert(ss>=0 && ss<Nm);
      fX[ss] += o->fX[i];
      fY[ss] += o->fY[i];
      tZ[ss] += (pX[ss]-o->pX[i])*o->fY[i] - (pY[ss]-o->pY[i])*o->fX[i];
    }
  }
  // 3) all reduce across ranks
  MPI_Allreduce(MPI_IN_PLACE, fX, Nm, MPI_DOUBLE, MPI_SUM, grid->getCartComm());
  MPI_Allreduce(MPI_IN_PLACE, fY, Nm, MPI_DOUBLE, MPI_SUM, grid->getCartComm());
  MPI_Allreduce(MPI_IN_PLACE, tZ, Nm, MPI_DOUBLE, MPI_SUM, grid->getCartComm());
}
