/*
 *  FDcoeffs.h
 *  CubismUP_3D
 *
 *  Created by Fabian Wermelinger 07/24/2017
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#ifndef CubismUP_3D_FDcoeffs_h
#define CubismUP_3D_FDcoeffs_h

#include <Cubism/MeshMap.h>

#include <cassert>
#include <cstdlib>

// helpers
///////////////////////////////////////////////////////////////////////////////
#define __FD_2ND(i,c,um1,u00,up1) (c.cm1[i]*um1 + c.c00[i]*u00 + c.cp1[i]*up1)
#define __FD_4TH(i,c,um2,um1,u00,up1,up2) (c.cm2[i]*um2 + c.cm1[i]*um1 + c.c00[i]*u00 + c.cp1[i]*up1 + c.cp2[i]*up2)

CubismUP_3D_NAMESPACE_BEGIN

// coefficient set
///////////////////////////////////////////////////////////////////////////////
// 2nd order scheme
template <size_t _SIZE_>
struct BlkCoeff_2ndOrder_t
{
    Real cm1[_SIZE_];
    Real c00[_SIZE_];
    Real cp1[_SIZE_];
};

// 4th order scheme
template <size_t _SIZE_>
struct BlkCoeff_4thOrder_t
{
    Real cm2[_SIZE_];
    Real cm1[_SIZE_];
    Real c00[_SIZE_];
    Real cp1[_SIZE_];
    Real cp2[_SIZE_];
};

typedef BlkCoeff_2ndOrder_t<CUP_BLOCK_SIZE> BlkCoeffX;
typedef BlkCoeff_2ndOrder_t<CUP_BLOCK_SIZE> BlkCoeffY;
typedef BlkCoeff_2ndOrder_t<CUP_BLOCK_SIZE> BlkCoeffZ;
typedef BlkCoeff_4thOrder_t<CUP_BLOCK_SIZE> BlkCoeffX_4;
typedef BlkCoeff_4thOrder_t<CUP_BLOCK_SIZE> BlkCoeffY_4;
typedef BlkCoeff_4thOrder_t<CUP_BLOCK_SIZE> BlkCoeffZ_4;
///////////////////////////////////////////////////////////////////////////////

// per block coefficients
struct FDBlockCoeffs_x
{
    // first and second derivative (2nd Order)
    BlkCoeffX first;
    BlkCoeffX second;

    // first and second derivative (4th Order)
    BlkCoeffX_4 first_4;
    BlkCoeffX_4 second_4;
};
struct FDBlockCoeffs_y
{
    // first and second derivative (2nd Order)
    BlkCoeffY first;
    BlkCoeffY second;

    // first and second derivative (4th Order)
    BlkCoeffY_4 first_4;
    BlkCoeffY_4 second_4;
};
struct FDBlockCoeffs_z
{
    // first and second derivative (2nd Order)
    BlkCoeffZ first;
    BlkCoeffZ second;

    // first and second derivative (4th Order)
    BlkCoeffZ_4 first_4;
    BlkCoeffZ_4 second_4;
};

class FDcoeffs_1DBase
{
public:
    FDcoeffs_1DBase(const unsigned int N) : m_initialized(false), m_N(N) {}
    virtual ~FDcoeffs_1DBase() {}
    virtual void setup(const double* const grid_spacing, const unsigned int ncells) = 0;

protected:
    bool m_initialized;
    const unsigned int m_N;
};


///////////////////////////////////////////////////////////////////////////////
// Implementation of 2nd order finite differences for 1st and 2nd derivative
// REMARK:
// The scheme for the 2nd derivative is technically 1st order accurate.
// However, this is only true on very distorted grids (e.g. random position of
// grid nodes).  The leading error term is proportional to
//
//    \epsilon \propto h_i^2 * (1 - \frac{h_{i-1}}{h_{i+1}})
//
// where h_i is the grid spacing.  On a smoothly streteched grid, h_{i-1} and
// h_{i+1} are well correlated and the leading error term cancels almost
// perfectly, giving an extra order of accuracy with the same stencil width.
// On a uniform mesh, this scheme is exactly second order accurate.
///////////////////////////////////////////////////////////////////////////////
class FDcoeffs_2ndOrder: public FDcoeffs_1DBase
{
  public:
    template <size_t _BSIZE>
    class BlockSetFunctor
    {
      public:
        template <typename T>
        void operator()(const FDcoeffs_2ndOrder& fd, T& bcoeffs, const unsigned int offset=0)
        {
            const Real * const c1_m1 = fd.first_cm1();
            const Real * const c1_00 = fd.first_c00();
            const Real * const c1_p1 = fd.first_cp1();
            const Real * const c2_m1 = fd.second_cm1();
            const Real * const c2_00 = fd.second_c00();
            const Real * const c2_p1 = fd.second_cp1();
            for (unsigned int i = 0; i < _BSIZE; ++i)
            {
                bcoeffs.first.cm1[i] = c1_m1[offset+i];
                bcoeffs.first.c00[i] = c1_00[offset+i];
                bcoeffs.first.cp1[i] = c1_p1[offset+i];
                bcoeffs.second.cm1[i] = c2_m1[offset+i];
                bcoeffs.second.c00[i] = c2_00[offset+i];
                bcoeffs.second.cp1[i] = c2_p1[offset+i];
            }
        }
    };

  public:
    FDcoeffs_2ndOrder(const unsigned int N) : FDcoeffs_1DBase(N) {}
    virtual ~FDcoeffs_2ndOrder() { if (m_initialized) _dealloc(); }

    static const unsigned int HALO_S = 1;
    static const unsigned int HALO_E = 1;

    inline const Real * first_cm1() const { return m_first_cm1; }
    inline const Real * first_c00() const { return m_first_c00; }
    inline const Real * first_cp1() const { return m_first_cp1; }
    inline const Real * second_cm1() const { return m_second_cm1; }
    inline const Real * second_c00() const { return m_second_c00; }
    inline const Real * second_cp1() const { return m_second_cp1; }

    virtual void setup(const double* const grid_spacing, const unsigned int ncells)
    {
        _alloc();

        assert(ncells == m_N);
        // Coefficients map to cell center x_{i} and correspond to grid values
        // as:
        // cm1[i]*u[i-1] + c00[i]*u[i] + cp1[i]*u[i+1]
        for (int i = 0; i < (int)m_N; ++i)
        {
            const double* const si = grid_spacing + i;
            const double hm1 = *(si-1);
            const double h00 = *si;
            const double hp1 = *(si+1);

            const double dm1 = 0.5*(hm1+h00);
            const double dp1 = 0.5*(h00+hp1);

            // first derivative
            m_first_cm1[i] = -dp1 / (dm1*(dm1+dp1));
            m_first_c00[i] = (-dm1+dp1) / (dm1*dp1);
            m_first_cp1[i] =  dm1 / (dp1*(dm1+dp1));

            // second derivative
            m_second_cm1[i] =  2.0/(dm1*(dm1+dp1));
            m_second_c00[i] = -2.0/(dm1*dp1);
            m_second_cp1[i] =  2.0/(dp1*(dm1+dp1));
        }
    }

protected:
    Real* m_first_cm1;
    Real* m_first_c00;
    Real* m_first_cp1;
    Real* m_second_cm1;
    Real* m_second_c00;
    Real* m_second_cp1;

    inline void _alloc()
    {
        posix_memalign((void**)&m_first_cm1,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_c00,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_cp1,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cm1, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_c00, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cp1, 32, m_N*sizeof(Real));
        m_initialized = true;
    }

    inline void _dealloc()
    {
        free(m_first_cm1);
        free(m_first_c00);
        free(m_first_cp1);
        free(m_second_cm1);
        free(m_second_c00);
        free(m_second_cp1);
    }
};

class FDcoeffs_2ndOrder_h3: public FDcoeffs_1DBase
{
  public:
    template <size_t _BSIZE>
    class BlockSetFunctor
    {
      public:
        template <typename T>
        void operator()(const FDcoeffs_2ndOrder_h3& fd, T& bcoeffs, const unsigned int offset=0)
        {
            const Real * const c1_m1 = fd.first_cm1();
            const Real * const c1_00 = fd.first_c00();
            const Real * const c1_p1 = fd.first_cp1();
            const Real * const c2_m1 = fd.second_cm1();
            const Real * const c2_00 = fd.second_c00();
            const Real * const c2_p1 = fd.second_cp1();
            for (unsigned int i = 0; i < _BSIZE; ++i)
            {
                bcoeffs.first.cm1[i] = c1_m1[offset+i];
                bcoeffs.first.c00[i] = c1_00[offset+i];
                bcoeffs.first.cp1[i] = c1_p1[offset+i];
                bcoeffs.second.cm1[i] = c2_m1[offset+i];
                bcoeffs.second.c00[i] = c2_00[offset+i];
                bcoeffs.second.cp1[i] = c2_p1[offset+i];
            }
        }
    };

  public:
    FDcoeffs_2ndOrder_h3(const unsigned int N) : FDcoeffs_1DBase(N) {}
    virtual ~FDcoeffs_2ndOrder_h3() { if (m_initialized) _dealloc(); }

    static const unsigned int HALO_S = 1;
    static const unsigned int HALO_E = 1;

    inline const Real * first_cm1() const { return m_first_cm1; }
    inline const Real * first_c00() const { return m_first_c00; }
    inline const Real * first_cp1() const { return m_first_cp1; }
    inline const Real * second_cm1() const { return m_second_cm1; }
    inline const Real * second_c00() const { return m_second_c00; }
    inline const Real * second_cp1() const { return m_second_cp1; }

    virtual void setup(const double* const grid_spacing, const unsigned int ncells)
    {
        _alloc();

        assert(ncells == m_N);
        // Coefficients map to cell center x_{i} and correspond to grid values
        // as:
        // cm1[i]*u[i-1] + c00[i]*u[i] + cp1[i]*u[i+1]
        for (int i = 0; i < (int)m_N; ++i)
        {
            const double* const si = grid_spacing + i;
            const double hm1 = *(si-1);
            const double h00 = *si;
            const double hp1 = *(si+1);

            const double dm1 = 0.5*(hm1+h00);
            const double dp1 = 0.5*(h00+hp1);

            // first derivative
            m_first_cm1[i] = -dp1 / (dm1*(dm1+dp1));
            m_first_c00[i] = (-dm1+dp1) / (dm1*dp1);
            m_first_cp1[i] =  dm1 / (dp1*(dm1+dp1));

            // second derivative
            m_second_cm1[i] =  2.0/(dm1*(dm1+dp1));
            m_second_c00[i] = -2.0/(dm1*dp1);
            m_second_cp1[i] =  2.0/(dp1*(dm1+dp1));
        }
    }

  protected:
    Real* m_first_cm1;
    Real* m_first_c00;
    Real* m_first_cp1;
    Real* m_second_cm1;
    Real* m_second_c00;
    Real* m_second_cp1;

    inline void _alloc()
    {
        posix_memalign((void**)&m_first_cm1,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_c00,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_cp1,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cm1, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_c00, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cp1, 32, m_N*sizeof(Real));
        m_initialized = true;
    }

    inline void _dealloc()
    {
        free(m_first_cm1);
        free(m_first_c00);
        free(m_first_cp1);
        free(m_second_cm1);
        free(m_second_c00);
        free(m_second_cp1);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Implementation of 4th order finite differences for 1st and 2nd derivative.
// REMARK:
// The scheme for the 2nd derivative is technically 3rd order accurate.  See
// remark above.
///////////////////////////////////////////////////////////////////////////////
class FDcoeffs_4thOrder: public FDcoeffs_1DBase
{
  public:
    template <size_t _BSIZE>
    class BlockSetFunctor
    {
    public:
        template <typename T>
        void operator()(const FDcoeffs_4thOrder& fd, T& bcoeffs, const unsigned int offset=0)
        {
            const Real * const c1_m2 = fd.first_cm2();
            const Real * const c1_m1 = fd.first_cm1();
            const Real * const c1_00 = fd.first_c00();
            const Real * const c1_p1 = fd.first_cp1();
            const Real * const c1_p2 = fd.first_cp2();
            const Real * const c2_m2 = fd.second_cm2();
            const Real * const c2_m1 = fd.second_cm1();
            const Real * const c2_00 = fd.second_c00();
            const Real * const c2_p1 = fd.second_cp1();
            const Real * const c2_p2 = fd.second_cp2();
            for (unsigned int i = 0; i < _BSIZE; ++i)
            {
                bcoeffs.first_4.cm2[i] = c1_m2[offset+i];
                bcoeffs.first_4.cm1[i] = c1_m1[offset+i];
                bcoeffs.first_4.c00[i] = c1_00[offset+i];
                bcoeffs.first_4.cp1[i] = c1_p1[offset+i];
                bcoeffs.first_4.cp2[i] = c1_p2[offset+i];
                bcoeffs.second_4.cm2[i] = c2_m2[offset+i];
                bcoeffs.second_4.cm1[i] = c2_m1[offset+i];
                bcoeffs.second_4.c00[i] = c2_00[offset+i];
                bcoeffs.second_4.cp1[i] = c2_p1[offset+i];
                bcoeffs.second_4.cp2[i] = c2_p2[offset+i];
            }
        }
    };

public:
    FDcoeffs_4thOrder(const unsigned int N) : FDcoeffs_1DBase(N) {}
    virtual ~FDcoeffs_4thOrder() { if (m_initialized) _dealloc(); }

    static const unsigned int HALO_S = 2;
    static const unsigned int HALO_E = 2;

    inline const Real * first_cm2() const { return m_first_cm2; }
    inline const Real * first_cm1() const { return m_first_cm1; }
    inline const Real * first_c00() const { return m_first_c00; }
    inline const Real * first_cp1() const { return m_first_cp1; }
    inline const Real * first_cp2() const { return m_first_cp2; }
    inline const Real * second_cm2() const { return m_second_cm2; }
    inline const Real * second_cm1() const { return m_second_cm1; }
    inline const Real * second_c00() const { return m_second_c00; }
    inline const Real * second_cp1() const { return m_second_cp1; }
    inline const Real * second_cp2() const { return m_second_cp2; }

    virtual void setup(const double* const grid_spacing, const unsigned int ncells)
    {
        _alloc();

        assert(ncells == m_N);
        // Coefficients map to cell center x_{i} and correspond to grid values
        // as:
        // cm2[i]*u[i-2] + cm1[i]*u[i-1] + c00[i]*u[i] + cp1[i]*u[i+1] + cp2[i]*u[i+2]
        for (int i = 0; i < (int)m_N; ++i)
        {
            const double* const si = grid_spacing + i;
            const double hm2 = *(si-2);
            const double hm1 = *(si-1);
            const double h00 = *si;
            const double hp1 = *(si+1);
            const double hp2 = *(si+2);

            const double dm2 = hm1 + 0.5*(hm2+h00);
            const double dm1 = 0.5*(hm1+h00);
            const double dp1 = 0.5*(h00+hp1);
            const double dp2 = hp1 + 0.5*(h00+hp2);

            // first derivative
            m_first_cm2[i] =  dm1*dp1*dp2 / (dm2*(dm2-dm1)*(dm2+dp1)*(dm2+dp2));
            m_first_cm1[i] = -dm2*dp1*dp2 / (dm1*(dm2-dm1)*(dm1+dp1)*(dm1+dp2));
            m_first_c00[i] = (-dm2*dm1*dp1 -dm2*dm1*dp2 +dm2*dp1*dp2 +dm1*dp1*dp2) / (dm2*dm1*dp1*dp2);
            m_first_cp1[i] = -dm2*dm1*dp2 / (dp1*(dm2+dp1)*(dm1+dp1)*(dp1-dp2));
            m_first_cp2[i] =  dm2*dm1*dp1 / (dp2*(dp1-dp2)*(dm2+dp2)*(dm1+dp2));

            // second derivative
            m_second_cm2[i] = -2.0*(dm1*dp1 + dm1*dp2 - dp1*dp2) / (dm2*(dm2-dm1)*(dm2+dp1)*(dm2+dp2));
            m_second_cm1[i] =  2.0*(dm2*dp1 + dm2*dp2 - dp1*dp2) / (dm1*(dm2-dm1)*(dm1+dp1)*(dm1+dp2));
            m_second_c00[i] =  2.0*(dm2*dm1 - dm2*dp1 - dm1*dp1 - dm2*dp2 - dm1*dp2 + dp1*dp2) / (dm2*dm1*dp1*dp2);
            m_second_cp1[i] =  2.0*(dm2*dm1 - dm2*dp2 - dm1*dp2) / (dp1*(dm2+dp1)*(dm1+dp1)*(dp1-dp2));
            m_second_cp2[i] = -2.0*(dm2*dm1 - dm2*dp1 - dm1*dp1) / (dp2*(dp1-dp2)*(dm2+dp2)*(dm1+dp2));
        }
    }

protected:
    Real* m_first_cm2;
    Real* m_first_cm1;
    Real* m_first_c00;
    Real* m_first_cp1;
    Real* m_first_cp2;
    Real* m_second_cm2;
    Real* m_second_cm1;
    Real* m_second_c00;
    Real* m_second_cp1;
    Real* m_second_cp2;

    inline void _alloc()
    {
        posix_memalign((void**)&m_first_cm2,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_cm1,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_c00,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_cp1,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_first_cp2,  32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cm2, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cm1, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_c00, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cp1, 32, m_N*sizeof(Real));
        posix_memalign((void**)&m_second_cp2, 32, m_N*sizeof(Real));
        m_initialized = true;
    }

    inline void _dealloc()
    {
        free(m_first_cm2);
        free(m_first_cm1);
        free(m_first_c00);
        free(m_first_cp1);
        free(m_first_cp2);
        free(m_second_cm2);
        free(m_second_cm1);
        free(m_second_c00);
        free(m_second_cp1);
        free(m_second_cp2);
    }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_FDcoeffs_h
