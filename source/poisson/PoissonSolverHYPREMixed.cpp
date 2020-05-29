//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PoissonSolverHYPREMixed.h"
#ifdef CUP_HYPRE
#ifndef CUP_SINGLE_PRECISION
#define MPIREAL MPI_DOUBLE
#else
#define MPIREAL MPI_FLOAT
#endif /* CUP_SINGLE_PRECISION */

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

void PoissonSolverMixed_HYPRE::solve()
{
  sim.startProfiler("HYPRE cub2rhs");
  _cub2fftw();
  sim.stopProfiler();

  if(bRankHoldsFixedDOF) {
    // set last corner such that last point has pressure pLast
    data[fixed_idx]  = coef_fixed_idx*pLast;
    // neighbours read value of corner from the RHS:
    data[fixed_m1z] -= coef_fixed_m1z*pLast; //fixed dof-1dz reads from RHS +1dz
    data[fixed_p1z] -= coef_fixed_p1z*pLast; //fixed dof+1dz reads from RHS -1dz
    data[fixed_m1y] -= coef_fixed_m1y*pLast; //fixed dof-1dy reads from RHS +1dy
    data[fixed_p1y] -= coef_fixed_p1y*pLast; //fixed dof+1dy reads from RHS -1dy
    data[fixed_m1x] -= coef_fixed_m1x*pLast; //fixed dof-1dx reads from RHS +1dx
    data[fixed_p1x] -= coef_fixed_p1x*pLast; //fixed dof+1dx reads from RHS -1dx
  }

  sim.startProfiler("HYPRE setBoxV");
  HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, data);
  sim.stopProfiler();

  sim.startProfiler("HYPRE solve");
  if (solver == "gmres")
    HYPRE_StructGMRESSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else if (solver == "smg")
    HYPRE_StructSMGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  else
    HYPRE_StructPCGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  sim.stopProfiler();

  sim.startProfiler("HYPRE getBoxV");
  HYPRE_StructVectorGetBoxValues(hypre_sol, ilower, iupper, data);
  sim.stopProfiler();

  sim.startProfiler("HYPRE mean0");
  {
    const Real avgP = sim.bUseStretchedGrid? computeAverage_nonUniform()
                                           : computeAverage();
    // Subtract average pressure from all gridpoints
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < data_size; i++) data[i] -= avgP;
    // Set this new mean-0 pressure as next guess
    HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, data);
    // Save pressure of a corner of the grid so that it can be imposed next time
    pLast = data[fixed_idx];
    if(sim.verbose) printf("Avg Pressure:%f\n", avgP);
  }
  sim.stopProfiler();
}

PoissonSolverMixed_HYPRE::PoissonSolverMixed_HYPRE(SimulationData&s) :
  PoissonSolver(s), solver("pcg")
{
  printf("Employing HYPRE-based Poisson solver with Dirichlet BCs. Rank %d pos {%d %d %d}\n", m_rank, peidx[0], peidx[1], peidx[2]);
  if(bRankHoldsFixedDOF)
    printf("Rank %d holds the fixed DOF!\n", m_rank);
  fflush(0);
  data = new Real[myN[0] * myN[1] * myN[2]];
  data_size = (size_t) myN[0] * (size_t) myN[1] * (size_t) myN[2];
  stridez = myN[1] * myN[0]; // slow
  stridey = myN[0];
  stridex = 1; // fast

  // Grid
  HYPRE_StructGridCreate(m_comm, 3, &hypre_grid);

  HYPRE_StructGridSetExtents(hypre_grid, ilower, iupper);

  //HYPRE_Int ghosts[3] = {2, 2, 2};
  //HYPRE_StructGridSetNumGhost(hypre_grid, ghosts);

  // if grid is periodic, this function takes the period
  // length... ie. the grid size.
  HYPRE_Int iPeriod[3] = {
    sim.BCx_flag == periodic ? iGridEnd[0]+1 : 0,
    sim.BCy_flag == periodic ? iGridEnd[1]+1 : 0,
    sim.BCz_flag == periodic ? iGridEnd[2]+1 : 0
  };
  HYPRE_StructGridSetPeriodic(hypre_grid, iPeriod);

  HYPRE_StructGridAssemble(hypre_grid);

  { // Stencil
    HYPRE_Int offsets[7][3] = {{ 0, 0, 0},
                               {-1, 0, 0}, { 1, 0, 0},
                               { 0,-1, 0}, { 0, 1, 0},
                               { 0, 0,-1}, { 0, 0, 1}};
    HYPRE_StructStencilCreate(3, 7, &hypre_stencil);
    for (int j = 0; j < 7; ++j)
      HYPRE_StructStencilSetElement(hypre_stencil, j, offsets[j]);
  }

  { // Matrix
    HYPRE_StructMatrixCreate(m_comm, hypre_grid, hypre_stencil, &hypre_mat);
    //HYPRE_StructMatrixSetSymmetric(hypre_mat, 1);
    HYPRE_StructMatrixInitialize(hypre_mat);

    // These indices must match to those in the offset array:
    HYPRE_Int inds[7] = {0, 1, 2, 3, 4, 5, 6};


    RowType* const vals = sim.bUseStretchedGrid ? prepareMat_nonUniform()
                                                : prepareMat();
    if(sim.BCx_flag != periodic && ilower[0] == 0) {
      #pragma omp parallel for schedule(static)
      for(size_t k=0; k<myN[2]; k++) for(size_t j=0; j<myN[1]; j++) {
        const auto idx = linaccess(0, j, k);
        vals[idx][0] += vals[idx][1]; vals[idx][1] = 0;
      }
    }
    if(sim.BCx_flag != periodic && iupper[0] == iGridEnd[0]) {
      #pragma omp parallel for schedule(static)
      for(size_t k=0; k<myN[2]; k++) for(size_t j=0; j<myN[1]; j++) {
        const auto idx = linaccess(myN[0]-1, j, k);
        vals[idx][0] += vals[idx][2]; vals[idx][2] = 0;
      }
    }

    if(sim.BCy_flag != periodic && ilower[1] == 0) {
      #pragma omp parallel for schedule(static)
      for(size_t k=0; k<myN[2]; k++) for(size_t i=0; i<myN[0]; i++) {
        const auto idx = linaccess(i, 0, k);
        vals[idx][0] += vals[idx][3]; vals[idx][3] = 0;
      }
    }
    if(sim.BCy_flag != periodic && iupper[1] == iGridEnd[1]) {
      #pragma omp parallel for schedule(static)
      for(size_t k=0; k<myN[2]; k++) for(size_t i=0; i<myN[0]; i++) {
        const auto idx = linaccess(i, myN[1]-1, k);
        vals[idx][0] += vals[idx][4]; vals[idx][4] = 0;
      }
    }

    if(sim.BCz_flag != periodic && ilower[2] == 0) {
      #pragma omp parallel for schedule(static)
      for(size_t j=0; j<myN[1]; j++) for(size_t i=0; i<myN[0]; i++) {
        const auto idx = linaccess(i, j, 0);
        vals[idx][0] += vals[idx][5]; vals[idx][5] = 0;
      }
    }
    if(sim.BCz_flag != periodic && iupper[2] == iGridEnd[2]) {
      #pragma omp parallel for schedule(static)
      for(size_t j=0; j<myN[1]; j++) for(size_t i=0; i<myN[0]; i++) {
        const auto idx = linaccess(i, j, myN[2]-1);
        vals[idx][0] += vals[idx][6]; vals[idx][6] = 0;
      }
    }

    if(bRankHoldsFixedDOF)
    {
      // set last corner such that last point has pressure pLast
      coef_fixed_idx = vals[fixed_idx][0];
      assert(std::fabs(coef_fixed_idx) > 1e-16);
      vals[fixed_idx][1] = 0; vals[fixed_idx][2] = 0;
      vals[fixed_idx][3] = 0; vals[fixed_idx][4] = 0;
      vals[fixed_idx][5] = 0; vals[fixed_idx][6] = 0;
      // neighbours read value of corner from the RHS:
      //fixed dof-1dz reads +1dz from RHS
      coef_fixed_m1z = vals[fixed_m1z][6]; vals[fixed_m1z][6] = 0;
      //fixed dof+1dz reads -1dz from RHS
      coef_fixed_p1z = vals[fixed_p1z][5]; vals[fixed_p1z][5] = 0;
      //fixed dof-1dy reads +1dy from RHS
      coef_fixed_m1y = vals[fixed_m1y][4]; vals[fixed_m1y][4] = 0;
      //fixed dof+1dy reads -1dy from RHS
      coef_fixed_p1y = vals[fixed_p1y][3]; vals[fixed_p1y][3] = 0;
      //fixed dof-1dx reads +1dx from RHS
      coef_fixed_m1x = vals[fixed_m1x][2]; vals[fixed_m1x][2] = 0;
      //fixed dof+1dx reads -1dx from RHS
      coef_fixed_p1x = vals[fixed_p1x][1]; vals[fixed_p1x][1] = 0;
    }

    //Real* const linV = static_cast<Real*> (& vals[0][0]);
    Real* const linV = reinterpret_cast<Real*> (vals);
    assert(linV not_eq nullptr);
    HYPRE_StructMatrixSetBoxValues(hypre_mat, ilower, iupper, 7, inds, linV);
    HYPRE_StructMatrixAssemble(hypre_mat);
    delete [] vals;
  }

  // Rhs and initial guess
  HYPRE_StructVectorCreate(m_comm, hypre_grid, &hypre_rhs);
  HYPRE_StructVectorCreate(m_comm, hypre_grid, &hypre_sol);

  HYPRE_StructVectorInitialize(hypre_rhs);
  HYPRE_StructVectorInitialize(hypre_sol);

  {
    memset(data, 0, myN[0] * myN[1] * myN[2] * sizeof(Real));
    HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, data);
    HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, data);
  }

  HYPRE_StructVectorAssemble(hypre_rhs);
  HYPRE_StructVectorAssemble(hypre_sol);

  if (solver == "gmres") {
    printf("Using GMRES solver\n"); fflush(0);
    HYPRE_StructGMRESCreate(m_comm, &hypre_solver);
    HYPRE_StructGMRESSetTol(hypre_solver, 1e-2);
    HYPRE_StructGMRESSetPrintLevel(hypre_solver, 2);
    HYPRE_StructGMRESSetMaxIter(hypre_solver, 1000);
    HYPRE_StructGMRESSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else if (solver == "smg") {
    printf("Using SMG solver\n"); fflush(0);
    HYPRE_StructSMGCreate(m_comm, &hypre_solver);
    //HYPRE_StructSMGSetMemoryUse(hypre_solver, 0);
    HYPRE_StructSMGSetMaxIter(hypre_solver, 100);
    HYPRE_StructSMGSetTol(hypre_solver, 1e-3);
    //HYPRE_StructSMGSetRelChange(hypre_solver, 0);
    HYPRE_StructSMGSetPrintLevel(hypre_solver, 3);
    HYPRE_StructSMGSetNumPreRelax(hypre_solver, 1);
    HYPRE_StructSMGSetNumPostRelax(hypre_solver, 1);

    HYPRE_StructSMGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
  else {
    printf("Using PCG solver\n"); fflush(0);
    HYPRE_StructPCGCreate(m_comm, &hypre_solver);
    HYPRE_StructPCGSetMaxIter(hypre_solver, 1000);
    HYPRE_StructPCGSetTol(hypre_solver, 1e-3);
    HYPRE_StructPCGSetPrintLevel(hypre_solver, 0);
    if(0)
    { // Use SMG preconditioner
      HYPRE_StructSMGCreate(m_comm, &hypre_precond);
      HYPRE_StructSMGSetMaxIter(hypre_precond, 1000);
      HYPRE_StructSMGSetTol(hypre_precond, 0);
      HYPRE_StructSMGSetNumPreRelax(hypre_precond, 1);
      HYPRE_StructSMGSetNumPostRelax(hypre_precond, 1);
      HYPRE_StructPCGSetPrecond(hypre_solver, HYPRE_StructSMGSolve,
                                HYPRE_StructSMGSetup, hypre_precond);
    }
    HYPRE_StructPCGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
  }
}

PoissonSolverMixed_HYPRE::~PoissonSolverMixed_HYPRE()
{
  if (solver == "gmres")
    HYPRE_StructGMRESDestroy(hypre_solver);
  else if (solver == "smg")
    HYPRE_StructSMGDestroy(hypre_solver);
  else
    HYPRE_StructPCGDestroy(hypre_solver);
  HYPRE_StructGridDestroy(hypre_grid);
  HYPRE_StructStencilDestroy(hypre_stencil);
  HYPRE_StructMatrixDestroy(hypre_mat);
  HYPRE_StructVectorDestroy(hypre_rhs);
  HYPRE_StructVectorDestroy(hypre_sol);
  delete [] data;
}

using RowType = PoissonSolverMixed_HYPRE::RowType;
RowType* PoissonSolverMixed_HYPRE::prepareMat_nonUniform()
{
  RowType * const vals = new RowType[myN[0] * myN[1] * myN[2]];
  const auto& vInfo = sim.vInfo();
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<local_infos.size(); ++i)
  {
    const size_t offset = _offset(local_infos[i]);
    const FluidBlock& b = *(FluidBlock*) local_infos[i].ptrBlock;
    const BlkCoeffX &cx=b.fd_cx.second, &cy=b.fd_cy.second, &cz=b.fd_cz.second;
    for(size_t iz=0; iz < (size_t) BlockType::sizeZ; iz++)
    for(size_t iy=0; iy < (size_t) BlockType::sizeY; iy++)
    for(size_t ix=0; ix < (size_t) BlockType::sizeX; ix++)
    {
      const size_t idx = _dest(offset, iz, iy, ix);
      assert(idx < myN[0] * myN[1] * myN[2]);
      Real vh[3]; vInfo[i].spacing(vh, ix, iy, iz);
      const Real dv = vh[0] * vh[1] * vh[2];
      vals[idx][0] = dv * ( cx.c00[ix] + cy.c00[iy] + cz.c00[iz] );
      vals[idx][1] = dv *   cx.cm1[ix]; /* west  */
      vals[idx][2] = dv *   cx.cp1[ix]; /* east  */
      vals[idx][3] = dv *   cy.cm1[iy]; /* south */
      vals[idx][4] = dv *   cy.cp1[iy]; /* north */
      vals[idx][5] = dv *   cz.cm1[iz]; /* front */
      vals[idx][6] = dv *   cz.cp1[iz]; /* back  */
    }
  }
  return vals;
}

RowType* PoissonSolverMixed_HYPRE::prepareMat()
{
  const double h = sim.uniformH();
  using RowType = Real[7];
  RowType * const vals = new RowType[myN[0] * myN[1] * myN[2]];
  #pragma omp parallel for schedule(static)
  for (size_t k = 0; k < myN[2]; k++)
  for (size_t j = 0; j < myN[1]; j++)
  for (size_t i = 0; i < myN[0]; i++) {
    const auto idx = linaccess(i, j, k);
    assert(idx < (size_t) myN[0] * myN[1] * myN[2]);
    vals[idx][0] = -6*h; /* center */
    vals[idx][1] =  1*h; /* west   */ vals[idx][2] =  1*h; /* east   */
    vals[idx][3] =  1*h; /* south  */ vals[idx][4] =  1*h; /* north  */
    vals[idx][5] =  1*h; /* front  */ vals[idx][6] =  1*h; /* back   */
  }
  return vals;
}

CubismUP_3D_NAMESPACE_END
#endif
