//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#ifdef CUP_PETSC
#include "PoissonSolverPETSCMixed.h"

//#define CORNERFIX
// ^ this ^ fixes one DOF of the grid in order to have mean 0 pressure
// seems to be unnecessary if we remove the nullspace of the matrix

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#ifndef CUP_SINGLE_PRECISION
#define MPIREAL MPI_DOUBLE
#else
#define MPIREAL MPI_FLOAT
#endif /* CUP_SINGLE_PRECISION */

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

extern PetscErrorCode ComputeRHS(KSP solver, Vec RHS, void* D);
extern PetscErrorCode ComputeMAT(KSP solver, Mat A, Mat P, void* D);
extern PetscErrorCode ComputeMAT_nonUniform(KSP solver, Mat A, Mat P, void* D);
std::vector<char*> readRunArgLst();

struct PoissonSolverMixed_PETSC::PetscData
{
  Real * const cupRHS;
  const size_t cupRHS_size;
  const int myNx, myNy, myNz;
  const int gNx, gNy, gNz;
  const Real h;
  const MPI_Comm m_comm;
  const BCflag BCx, BCy, BCz;
  const std::vector<cubism::BlockInfo>& local_infos;
  int rank;
  KSP solver;
  DM grid;
  Vec SOL;

  PetscData(MPI_Comm c, size_t gx, size_t gy, size_t gz, size_t nx, size_t ny,
  size_t nz, size_t nsize, BCflag BCX, BCflag BCY, BCflag BCZ, Real H, Real*ptr,
  const std::vector<cubism::BlockInfo>& infos)
  : cupRHS(ptr), cupRHS_size(nsize), myNx(nx), myNy(ny), myNz(nz), gNx(gx),
  gNy(gy), gNz(gz), h(H), m_comm(c), BCx(BCX), BCy(BCY), BCz(BCZ), local_infos(infos) { }
  ~PetscData() {
    VecDestroy(& SOL);
    DMDestroy(& grid);
    KSPDestroy(& solver);
  }
};

PoissonSolverMixed_PETSC::PoissonSolverMixed_PETSC(SimulationData&s) : PoissonSolver(s)
{
  PETSC_COMM_WORLD = m_comm;
  data = new Real[myN[0] * myN[1] * myN[2]];
  data_size = (size_t) myN[0] * (size_t) myN[1] * (size_t) myN[2];
  stridez = myN[1] * myN[0]; // slow
  stridey = myN[0];
  stridex = 1; // fast
  const double h = sim.bUseStretchedGrid? -1 : sim.uniformH();
  S= new PetscData(m_comm, gsize[0], gsize[1], gsize[2], myN[0], myN[1], myN[2],
    data_size, sim.BCx_flag, sim.BCy_flag, sim.BCz_flag, h, data, local_infos);
  S->rank = m_rank;
  // (int argc,char **argv)
  std::vector<char*> args = readRunArgLst();
  int argc = args.size()-1; char ** argv = args.data();

  PetscInitialize(&argc, &argv, (char*)0, (char*)0);
  PetscErrorCode ierr = KSPCreate(m_comm, & S->solver);
  ierr = KSPSetFromOptions(S->solver);
  ierr = DMDACreate3d(m_comm,
    sim.BCx_flag == periodic ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
    sim.BCy_flag == periodic ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
    sim.BCz_flag == periodic ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE,
    DMDA_STENCIL_STAR, gsize[0], gsize[1], gsize[2],
    sim.nprocsx, sim.nprocsy, sim.nprocsz, 1, 1, NULL, NULL, NULL, & S->grid);
  ierr = DMSetFromOptions(S->grid);
  ierr = DMSetUp(S->grid);
  ierr = KSPSetDM(S->solver, S->grid);

  if(ierr) std::cout << "PETSC ERROR ID " << ierr << std::endl;
  DMSetApplicationContext(S->grid, S);
  KSPSetComputeRHS(S->solver, ComputeRHS, S);
  if(sim.bUseStretchedGrid)
    KSPSetComputeOperators(S->solver, ComputeMAT_nonUniform, S);
  else KSPSetComputeOperators(S->solver, ComputeMAT, S);
  KSPSetInitialGuessNonzero(S->solver, PETSC_TRUE );
}

PoissonSolverMixed_PETSC::~PoissonSolverMixed_PETSC()
{
  delete S;
  PetscFinalize();
}

void PoissonSolverMixed_PETSC::solve()
{
  sim.startProfiler("PETSC cub2rhs");
  _cub2fftw();
  sim.stopProfiler();

  sim.startProfiler("PETSC solve");
  KSPSolve(S->solver, NULL, NULL);
  sim.stopProfiler();

  {
    KSPGetSolution(S->solver, & S->SOL);
    PetscScalar ***array;
    DMDAVecGetArray(S->grid, S->SOL, & array);
    PetscInt x0, y0, z0, xN, yN, zN;
    DMDAGetCorners(S->grid, &x0, &y0, &z0, &xN, &yN, &zN);
    assert(myN[0]==(size_t)xN && myN[1]==(size_t)yN && myN[2]==(size_t)zN);
    sim.startProfiler("PETSC mean0");
    {
      Real avgP = 0;
      const Real fac = 1.0 / (gsize[0] * gsize[1] * gsize[2]);
      // Compute average pressure across all ranks:
      #pragma omp parallel for schedule(static) reduction(+ : avgP)
      for(int k=0;k<zN;k++) for(int j=0;j<yN;j++) for(int i=0;i<xN;i++)
        avgP += fac * array[k+z0][j+y0][i+x0];
      MPI_Allreduce(MPI_IN_PLACE, &avgP, 1, MPIREAL, MPI_SUM, m_comm);

      // Subtract average pressure from all gridpoints
      #pragma omp parallel for schedule(static)
      for(int k=0;k<zN;k++) for(int j=0;j<yN;j++) for(int i=0;i<xN;i++) {
        array[k+z0][j+y0][i+x0] -= avgP;
        data[stridex*i + stridey*j + stridez*k] = array[k+z0][j+y0][i+x0];
      }
      // Save pres of a corner of the grid so that it can be imposed next time
      pLast = data[fixed_idx];

      DMDAVecRestoreArray(S->grid, S->SOL, & array);
      VecAssemblyBegin(S->SOL);
      VecAssemblyEnd(S->SOL);
    }
    sim.stopProfiler();
  }
}

PetscErrorCode ComputeRHS(KSP solver, Vec RHS, void * Sptr)
{
  const auto& S = *( PoissonSolverMixed_PETSC::PetscData *) Sptr;
  const Real* const CubRHS = S.cupRHS;
  const size_t cSx = S.myNx, cSy = S.myNy;
  PetscScalar * * * array;
  PetscInt xStart, yStart, zStart, xSpan, ySpan, zSpan;
  DMDAGetCorners(S.grid, &xStart,&yStart,&zStart, &xSpan,&ySpan,&zSpan);
  DMDAVecGetArray(S.grid, RHS, & array);
  #pragma omp parallel for schedule(static)
  for (PetscInt k=0; k<zSpan; k++)
  for (PetscInt j=0; j<ySpan; j++)
  for (PetscInt i=0; i<xSpan; i++)
    array[k+zStart][j+yStart][i+xStart] = CubRHS[i + cSx*j + cSx*cSy*k];

  DMDAVecRestoreArray(S.grid, RHS, & array);
  VecAssemblyBegin(RHS);
  VecAssemblyEnd(RHS);
  // force right hand side to be consistent for singular matrix
  // it is just a hack that we avoid by fixing one DOF
  #if 1
    MatNullSpace nullspace;
    MatNullSpaceCreate(S.m_comm, PETSC_TRUE, 0, 0, &nullspace);
    MatNullSpaceRemove(nullspace, RHS);
    MatNullSpaceDestroy(&nullspace);
  #endif
  return 0;
}

PetscErrorCode ComputeMAT(KSP solver, Mat AMAT, Mat PMAT, void *Sptr)
{
  const auto& S = *( PoissonSolverMixed_PETSC::PetscData *) Sptr;
  printf("ComputeJacobian %d\n", S.rank); fflush(0);
  PetscInt xSt, ySt, zSt, xSpan, ySpan, zSpan;
  const Real h = S.h;
  DMDAGetCorners(S.grid, &xSt,&ySt,&zSt, &xSpan,&ySpan,&zSpan);
  #pragma omp parallel for schedule(static)
  for (int k=0; k<zSpan; k++)
  for (int j=0; j<ySpan; j++)
  for (int i=0; i<xSpan; i++)
  {
    const int I = xSt+i, J = ySt+j, K = zSt+k;
    MatStencil R, C[7];
    R.k = K; R.j = J; R.i = I;
    PetscScalar V[7];
    V[0] = -6*h; C[0].i = I;   C[0].j = J;   C[0].k = K;
    V[1] =    h; C[1].i = I-1; C[1].j = J;   C[1].k = K;
    V[2] =    h; C[2].i = I+1; C[2].j = J;   C[2].k = K;
    V[3] =    h; C[3].i = I;   C[3].j = J-1; C[3].k = K;
    V[4] =    h; C[4].i = I;   C[4].j = J+1; C[4].k = K;
    V[5] =    h; C[5].i = I;   C[5].j = J;   C[5].k = K-1;
    V[6] =    h; C[6].i = I;   C[6].j = J;   C[6].k = K+1;
    // Apply dirichlet BC:
    if( S.BCz != periodic && K == S.gNz-1 ) { V[0] += V[6]; V[6] = 0; }
    if( S.BCz != periodic && K ==       0 ) { V[0] += V[5]; V[5] = 0; }
    if( S.BCy != periodic && J == S.gNy-1 ) { V[0] += V[4]; V[4] = 0; }
    if( S.BCy != periodic && J ==       0 ) { V[0] += V[3]; V[3] = 0; }
    if( S.BCx != periodic && I == S.gNx-1 ) { V[0] += V[2]; V[2] = 0; }
    if( S.BCx != periodic && I ==       0 ) { V[0] += V[1]; V[1] = 0; }

    int nFilled = 7;
    if( std::fabs(V[6]) < 1e-16 ) { nFilled--;  // row 6 is unneeded
    }
    if( std::fabs(V[5]) < 1e-16 ) { nFilled--;  // row 5 is unneeded
      V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k; // 6 to 5
    }
    if( std::fabs(V[4]) < 1e-16 ) { nFilled--;  // row 4 is unneeded
      V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k; // 5 to 4
      V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k; // 6 to 5
    }
    if( std::fabs(V[3]) < 1e-16 ) { nFilled--;  // row 3 is unneeded
      V[3] = V[4]; C[3].i = C[4].i; C[3].j = C[4].j; C[3].k = C[4].k; // 4 to 3
      V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k; // 5 to 4
      V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k; // 6 to 5
    }
    if( std::fabs(V[2]) < 1e-16 ) { nFilled--;  // row 2 is unneeded
      V[2] = V[3]; C[2].i = C[3].i; C[2].j = C[3].j; C[2].k = C[3].k; // 3 to 2
      V[3] = V[4]; C[3].i = C[4].i; C[3].j = C[4].j; C[3].k = C[4].k; // 4 to 3
      V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k; // 5 to 4
      V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k; // 6 to 5
    }
    if( std::fabs(V[1]) < 1e-16 ) { nFilled--;  // row 1 is unneeded
      V[1] = V[2]; C[1].i = C[2].i; C[1].j = C[2].j; C[1].k = C[2].k; // 2 to 1
      V[2] = V[3]; C[2].i = C[3].i; C[2].j = C[3].j; C[2].k = C[3].k; // 3 to 2
      V[3] = V[4]; C[3].i = C[4].i; C[3].j = C[4].j; C[3].k = C[4].k; // 4 to 3
      V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k; // 5 to 4
      V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k; // 6 to 5
    }
    //MatSetValuesStencil(AMAT, 1, &R, nFilled, C, V, INSERT_VALUES);
    MatSetValuesStencil(PMAT, 1, &R, nFilled, C, V, INSERT_VALUES);
  }
  //MatAssemblyBegin(AMAT, MAT_FINAL_ASSEMBLY);
  //MatAssemblyEnd(AMAT, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(PMAT, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(PMAT, MAT_FINAL_ASSEMBLY);
  #if 1
    MatNullSpace nullspace;
    MatNullSpaceCreate(S.m_comm, PETSC_TRUE, 0, 0, &nullspace);
    MatSetNullSpace(AMAT, nullspace);
    MatNullSpaceDestroy(&nullspace);
  #endif
  return 0;
}

PetscErrorCode ComputeMAT_nonUniform(KSP solver, Mat AMAT, Mat PMAT, void *Sptr)
{
  const auto& S = *( PoissonSolverMixed_PETSC::PetscData *) Sptr;
  printf("ComputeJacobian_nonUniform %d\n", S.rank); fflush(0);
  PetscInt xSt, ySt, zSt, xSpan, ySpan, zSpan;
  DMDAGetCorners(S.grid, &xSt,&ySt,&zSt, &xSpan,&ySpan,&zSpan);
  assert(zSpan==S.myNz && ySpan==S.myNy && xSpan==S.myNx);
  static constexpr int BS[3] = {CUP_BLOCK_SIZE, CUP_BLOCK_SIZE, CUP_BLOCK_SIZE};

  const auto& local_infos = S.local_infos;
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<local_infos.size(); ++i)
  {
    const int bIstart[3] = {
      local_infos[i].index[0] * BS[0],
      local_infos[i].index[1] * BS[1],
      local_infos[i].index[2] * BS[2]
    };

    const FluidBlock& b = *(FluidBlock*) local_infos[i].ptrBlock;
    const BlkCoeffX &cx=b.fd_cx.second, &cy=b.fd_cy.second, &cz=b.fd_cz.second;

    for(size_t iz=0; iz < (size_t) cubismup3d::FluidBlock::sizeZ; iz++)
    for(size_t iy=0; iy < (size_t) cubismup3d::FluidBlock::sizeY; iy++)
    for(size_t ix=0; ix < (size_t) cubismup3d::FluidBlock::sizeX; ix++)
    {
      const int I = xSt + bIstart[0] + iz;
      const int J = ySt + bIstart[1] + iy;
      const int K = zSt + bIstart[2] + ix;
      MatStencil R, C[7];
      R.k = K; R.j = J; R.i = I;
      PetscScalar V[7];

      Real vh[3]; local_infos[i].spacing(vh, ix, iy, iz);
      const Real dv = vh[0] * vh[1] * vh[2];

      V[0] = dv * ( cx.c00[ix] + cy.c00[iy] + cz.c00[iz] );
      C[0].i = I;   C[0].j = J;   C[0].k = K;

      V[1] = dv *   cx.cm1[ix]; /* west  */
      C[1].i = I-1; C[1].j = J;   C[1].k = K;

      V[2] = dv *   cx.cp1[ix]; /* east  */
      C[2].i = I+1; C[2].j = J;   C[2].k = K;

      V[3] = dv *   cy.cm1[iy]; /* south */
      C[3].i = I;   C[3].j = J-1; C[3].k = K;

      V[4] = dv *   cy.cp1[iy]; /* north */
      C[4].i = I;   C[4].j = J+1; C[4].k = K;

      V[5] = dv *   cz.cm1[iz]; /* front */
      C[5].i = I;   C[5].j = J;   C[5].k = K-1;

      V[6] = dv *   cz.cp1[iz]; /* back  */
      C[6].i = I;   C[6].j = J;   C[6].k = K+1;

      // Apply dirichlet BC:
      if( S.BCz != periodic && K == S.gNz-1 ) { V[0] += V[6]; V[6] = 0; }
      if( S.BCz != periodic && K ==       0 ) { V[0] += V[5]; V[5] = 0; }
      if( S.BCy != periodic && J == S.gNy-1 ) { V[0] += V[4]; V[4] = 0; }
      if( S.BCy != periodic && J ==       0 ) { V[0] += V[3]; V[3] = 0; }
      if( S.BCx != periodic && I == S.gNx-1 ) { V[0] += V[2]; V[2] = 0; }
      if( S.BCx != periodic && I ==       0 ) { V[0] += V[1]; V[1] = 0; }

      int nFilled = 7;
      if( std::fabs(V[6]) < 1e-16 ) { nFilled--;  // row 6 is unneeded
      }
      if( std::fabs(V[5]) < 1e-16 ) { nFilled--;  // row 5 is unneeded
        V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k;// 6 to 5
      }
      if( std::fabs(V[4]) < 1e-16 ) { nFilled--;  // row 4 is unneeded
        V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k;// 5 to 4
        V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k;// 6 to 5
      }
      if( std::fabs(V[3]) < 1e-16 ) { nFilled--;  // row 3 is unneeded
        V[3] = V[4]; C[3].i = C[4].i; C[3].j = C[4].j; C[3].k = C[4].k;// 4 to 3
        V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k;// 5 to 4
        V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k;// 6 to 5
      }
      if( std::fabs(V[2]) < 1e-16 ) { nFilled--;  // row 2 is unneeded
        V[2] = V[3]; C[2].i = C[3].i; C[2].j = C[3].j; C[2].k = C[3].k;// 3 to 2
        V[3] = V[4]; C[3].i = C[4].i; C[3].j = C[4].j; C[3].k = C[4].k;// 4 to 3
        V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k;// 5 to 4
        V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k;// 6 to 5
      }
      if( std::fabs(V[1]) < 1e-16 ) { nFilled--;  // row 1 is unneeded
        V[1] = V[2]; C[1].i = C[2].i; C[1].j = C[2].j; C[1].k = C[2].k;// 2 to 1
        V[2] = V[3]; C[2].i = C[3].i; C[2].j = C[3].j; C[2].k = C[3].k;// 3 to 2
        V[3] = V[4]; C[3].i = C[4].i; C[3].j = C[4].j; C[3].k = C[4].k;// 4 to 3
        V[4] = V[5]; C[4].i = C[5].i; C[4].j = C[5].j; C[4].k = C[5].k;// 5 to 4
        V[5] = V[6]; C[5].i = C[6].i; C[5].j = C[6].j; C[5].k = C[6].k;// 6 to 5
      }
      MatSetValuesStencil(AMAT, 1, &R, nFilled, C, V, INSERT_VALUES);
      //MatSetValuesStencil(PMAT, 1, &R, nFilled, C, V, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(AMAT, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(AMAT, MAT_FINAL_ASSEMBLY);
  //MatAssemblyBegin(PMAT, MAT_FINAL_ASSEMBLY);
  //MatAssemblyEnd(PMAT, MAT_FINAL_ASSEMBLY);
  #if 1
    MatNullSpace nullspace;
    MatNullSpaceCreate(S.m_comm, PETSC_TRUE, 0, 0, &nullspace);
    MatSetNullSpace(AMAT, nullspace);
    MatNullSpaceDestroy(&nullspace);
  #endif
  return 0;
}

std::vector<char*> readRunArgLst()
{
  std::vector<char*> args;
  std::vector<std::string> params;
  //params.push_back("-ksp_monitor_short");

  // ksp seems to be the solver
  // Two optimized versions of conf gradient:
  params.push_back("-ksp_type"); params.push_back("cg");
  params.push_back("-ksp_cg_single_reduction"); // why not!
  //params.push_back("-ksp_type"); params.push_back("cgne"); // why not!
  //params.push_back("-ksp_type"); params.push_back("pipecgrr"); // why not!
  params.push_back("-ksp_rtol"); params.push_back("1e-3");
  params.push_back("-ksp_atol"); params.push_back("1e-4");

  // pc seems to be the preconditioner
  // quotes from the manual in comments:
  //PETSc provides fully supported (smoothed) aggregation AMG:
  params.push_back("-pc_type"); params.push_back("gamg");
  params.push_back("-pc_gamg_type"); params.push_back("agg");
  params.push_back("-pc_gamg_agg_nsmooths"); params.push_back("1");
  params.push_back("-pc_gamg_reuse_interpolation"); params.push_back("true");
  params.push_back("-mg_coarse_sub_pc_type"); params.push_back("svd");
  //params.push_back("-mg_levels_ksp_type"); params.push_back("richardson");
  params.push_back("-mg_levels_ksp_type"); params.push_back("chebyshev");
  params.push_back("-mg_levels_pc_type"); params.push_back("sor");
  params.push_back("-mg_levels_ksp_rtol"); params.push_back("1e-3");
  //Smoothed aggregation is recommended for symmetric positive defiite systems
  //The parameters for the eigen estimator can be set with the prefix gamg_est.
  //For example CG is a much better KSP type than the default GMRES if your
  //problem is symmetric positive definite;
  //params.push_back("-gamg_est_ksp_type"); params.push_back("cg");
  // ??? prepending any solver prefix that has been added to the solver ???

  for(size_t i=0; i<params.size(); i++)
  {
    char *arg = new char[params[i].size() + 1];
    copy(params[i].begin(), params[i].end(), arg);  // write into char array
    arg[params[i].size()] = '\0';
    args.push_back(arg);
  }
  args.push_back(0); // push back nullptr as last entry
  return args; // remember to deallocate it!
}

CubismUP_3D_NAMESPACE_END
#endif
