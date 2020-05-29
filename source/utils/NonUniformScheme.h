/*
 *  NonUniformScheme.h
 *  CubismUP_3D
 *
 *  Created by Fabian Wermelinger 05/08/2017
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#ifndef CubismUP_3D_NonUniformScheme_h
#define CubismUP_3D_NonUniformScheme_h

#include "../Definitions.h"

#include <Cubism/BlockInfo.h>
#include <Cubism/MeshMap.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

CubismUP_3D_NAMESPACE_BEGIN

template <typename TBlock>
class NonUniformScheme
{
    static constexpr size_t StencilMax = 3;

public:
    NonUniformScheme(
            const double xS, const double xE,
            const double yS, const double yE,
            const double zS, const double zE,
            const unsigned int nBlocksX, const unsigned int nBlocksY,
            const unsigned int nBlocksZ) :
        m_h_min{HUGE_VAL,HUGE_VAL,HUGE_VAL},  m_h_max{-1,-1,-1},
        m_initialized(false),
        m_map_x(xS,xE,nBlocksX),
        m_map_y(yS,yE,nBlocksY),
        m_map_z(zS,zE,nBlocksZ)
    {}
    ~NonUniformScheme() {}

    typedef cubism::MeshMap<TBlock> TMeshMap;

    template <int _S, int _E>
    class HaloVector
    {
      public:
        static const int START = _S;
        static const int END   = _E;

        inline double operator()(const int i) const { return m_data[i+_S]; }
        inline double& operator()(const int i) { return m_data[i+_S]; }
        inline void clear() { std::vector<double>().swap(m_data); }
        inline void fill(TMeshMap& mmap, const double* const halos)
        {
            m_data.resize(mmap.ncells() + _S + _E);
            m_data.insert(m_data.begin(), &halos[0], &halos[_S]);
            m_data.insert(m_data.begin()+_S, mmap.data_grid_spacing(), mmap.data_grid_spacing()+mmap.ncells());
            m_data.insert(m_data.begin()+_S+mmap.ncells(), &halos[_S], &halos[_S+_E]);
        }

    private:
        std::vector<double> m_data;
    };

    typedef HaloVector<StencilMax,StencilMax> TVector;


    void init(const cubism::MeshDensity* const kernel_x,
              const cubism::MeshDensity* const kernel_y,
              const cubism::MeshDensity* const kernel_z)
    {
        double ghosts[2*StencilMax];

        m_map_x.init(kernel_x, StencilMax, StencilMax, &ghosts[0]);
        m_all_delta_x.fill(m_map_x, &ghosts[0]);

        m_map_y.init(kernel_y, StencilMax, StencilMax, &ghosts[0]);
        m_all_delta_y.fill(m_map_y, &ghosts[0]);

        m_map_z.init(kernel_z, StencilMax, StencilMax, &ghosts[0]);
        m_all_delta_z.fill(m_map_z, &ghosts[0]);

        for (size_t i = 0; i < m_map_x.ncells(); ++i) {
          m_h_max[0] = std::max( m_h_max[0], m_map_x.cell_width(i) );
          m_h_min[0] = std::min( m_h_min[0], m_map_x.cell_width(i) );
          //if(m_map_x.cell_width(i)>m_h_max[0]) m_h_max[0]=m_map_x.cell_width(i);
          //if(m_map_x.cell_width(i)<m_h_min[0]) m_h_min[0]=m_map_x.cell_width(i);
        }

        for (size_t i = 0; i < m_map_y.ncells(); ++i) {
          m_h_max[1] = std::max( m_h_max[1], m_map_y.cell_width(i) );
          m_h_min[1] = std::min( m_h_min[1], m_map_y.cell_width(i) );
          //if(m_map_y.cell_width(i)>m_h_max[1]) m_h_max[1]=m_map_y.cell_width(i);
          //if(m_map_y.cell_width(i)<m_h_min[1]) m_h_min[1]=m_map_y.cell_width(i);
        }

        for (size_t i = 0; i < m_map_z.ncells(); ++i) {
          m_h_max[2] = std::max( m_h_max[2], m_map_z.cell_width(i) );
          m_h_min[2] = std::min( m_h_min[2], m_map_z.cell_width(i) );
          //if(m_map_z.cell_width(i)>m_h_max[2]) m_h_max[2]=m_map_z.cell_width(i);
          //if(m_map_z.cell_width(i)<m_h_min[2]) m_h_min[2]=m_map_z.cell_width(i);
        }
        assert( m_h_max[0]>=m_h_min[0] );
        assert( m_h_max[1]>=m_h_min[1] );
        assert( m_h_max[2]>=m_h_min[2] );
        m_initialized = true;
    }


    template <typename TFD>
    void setup_coefficients(std::vector<cubism::BlockInfo>& infos,
                            const bool cleanup = false)
    {
        if (!m_initialized)
        {
            fprintf(stderr,"ERROR: NonUniformScheme: Not initialized.\n");
            fflush(0); exit(1);
        }

        // 0. some checks
        // 1. compute coefficients for scheme TFD
        // 2. distribute coefficients over blocks
        // 3. cleanup (optional, if multiple schemes are setup, cleanup only at
        //    the end of initializing the last scheme.)

        // 0.
        assert(TFD::HALO_S <= StencilMax);
        assert(TFD::HALO_E <= StencilMax);

        // 1. non-uniform finite differences
        TFD fd_x(m_map_x.ncells());
        TFD fd_y(m_map_y.ncells());
        TFD fd_z(m_map_z.ncells());
        fd_x.setup(&m_all_delta_x(0), m_map_x.ncells());
        fd_y.setup(&m_all_delta_y(0), m_map_y.ncells());
        fd_z.setup(&m_all_delta_z(0), m_map_z.ncells());

        typename TFD::template BlockSetFunctor<CUP_BLOCK_SIZE> set_x;
        typename TFD::template BlockSetFunctor<CUP_BLOCK_SIZE> set_y;
        typename TFD::template BlockSetFunctor<CUP_BLOCK_SIZE> set_z;

        // 2.
#pragma omp parallel for
        for(size_t i=0; i<infos.size(); ++i)
        {
            cubism::BlockInfo info = infos[i];
            TBlock& b = *(TBlock*)info.ptrBlock;

            {
                const int index = info.index[0];
                const unsigned int offset = TBlock::sizeX * index;
                set_x(fd_x, b.fd_cx, offset);
            }
            {
                const int index = info.index[1];
                const unsigned int offset = TBlock::sizeY * index;
                set_y(fd_y, b.fd_cy, offset);
            }
            {
                const int index = info.index[2];
                const unsigned int offset = TBlock::sizeZ * index;
                set_z(fd_z, b.fd_cz, offset);
            }
        }

        // 3.
        if (cleanup)
        {
            m_all_delta_x.clear();
            m_all_delta_y.clear();
            m_all_delta_z.clear();
        }
    }

    // TODO: [fabianw@mavt.ethz.ch; Mon Jan 22 2018 07:44:01 PM (+0100)] Is this
    /*
    void setup_inverse_spacing(std::vector<cubism::BlockInfo>& infos)
    {
      if (!m_initialized)
      {
        fprintf(stderr,"ERROR: NonUniformScheme: Not initialized.\n");
        fflush(0); exit(1);
      }

      #pragma omp parallel for
      for(int i=0; i<(int)infos.size(); ++i)
      {
        cubism::BlockInfo info = infos[i];
        TBlock& b = *(TBlock*)info.ptrBlock;

        const int indx = info.index[0];
        _set_block_invh<CUP_BLOCK_SIZE>(m_map_x.get_grid_spacing(indx), &b.invh_x[0]);

        const int indy = info.index[1];
        _set_block_invh<CUP_BLOCK_SIZE>(m_map_y.get_grid_spacing(indy), &b.invh_y[0]);

        const int indy = info.index[2];
        _set_block_invh<CUP_BLOCK_SIZE>(m_map_z.get_grid_spacing(indz), &b.invh_z[0]);
      }
    }
    */

    inline const TMeshMap& get_map_x() const { return m_map_x; }
    inline const TMeshMap& get_map_y() const { return m_map_y; }
    inline const TMeshMap& get_map_z() const { return m_map_z; }
    inline TMeshMap& get_map_x() { return m_map_x; }
    inline TMeshMap& get_map_y() { return m_map_y; }
    inline TMeshMap& get_map_z() { return m_map_z; }

    inline double minimum_cell_width(const int i=-1) const
    {
        assert(i < 3 && i > -2);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: NonUniformScheme.h: minimum_cell_width() "
                            "can not return m_h_min, not initialized.\n");
            fflush(0); exit(1);
        }
        const double all_min = std::min({m_h_min[0], m_h_min[1], m_h_min[2]});
        if (-1 == i) return all_min;
        else         return m_h_min[i];
    }
    inline double maximum_cell_width(const int i=-1) const
    {
        assert(i < 3 && i > -2);
        if (!m_initialized)
        {
            fprintf(stderr, "ERROR: NonUniformScheme.h: maximum_cell_width() "
                            "can not return m_h_max, not initialized.\n");
            fflush(0); exit(1);
        }
        const double all_max = std::max({m_h_max[0], m_h_max[1], m_h_max[2]});
        if (-1 == i) return all_max;
        else         return m_h_max[i];
    }

    void print_mesh_statistics(const bool verb=true)
    {
        if (verb)
        {
            _compute_mesh_stats("x-direction", m_map_x.kernel_name(),
                       m_map_x.data_grid_spacing(), m_map_x.ncells() );
            _compute_mesh_stats("y-direction", m_map_y.kernel_name(),
                       m_map_y.data_grid_spacing(), m_map_y.ncells() );
            _compute_mesh_stats("z-direction", m_map_z.kernel_name(),
                       m_map_z.data_grid_spacing(), m_map_z.ncells() );
        }
    }

    double compute_mean_grid_spacing()
    {
      const double avgHx = _harmonicMean(m_map_x.data_grid_spacing(),
                                         m_map_x.ncells());
      const double avgHy = _harmonicMean(m_map_y.data_grid_spacing(),
                                         m_map_y.ncells());
      const double avgHz = _harmonicMean(m_map_z.data_grid_spacing(),
                                         m_map_z.ncells());
      //return ( avgHx + avgHy + avgHz ) / 3;
      return 3 / (1/avgHx + 1/avgHy + 1/avgHz);
    }

    double _harmonicMean(const double* const data, const unsigned int N)
    {
      double hmean = 0;
      for (unsigned int i = 0; i < N; ++i) hmean += 1.0 / data[i];
      return N / hmean;
    }

private:
    double m_h_min[3], m_h_max[3];
    bool m_initialized;
    TMeshMap m_map_x;
    TMeshMap m_map_y;
    TMeshMap m_map_z;

    TVector m_all_delta_x;
    TVector m_all_delta_y;
    TVector m_all_delta_z;

    template <int _BSIZE>
    void _set_block_invh(const double* const grid_spacing, Real* const invh)
    {
        for (int i = 0; i < _BSIZE; ++i)
            invh[i] = 1.0/grid_spacing[i];
    }

    void _compute_mesh_stats(const std::string header, const std::string name,
                             const double* const data, const unsigned int N)
    {
        const auto deltaM2 = [](double i, double delta) {
          return std::pow(delta, 2) * i / (i+1.0);
        };
        const auto deltaM3 = [](double i, double delta, double M2) {
          const double norm = i * (i-1.0) / std::pow(i+1.0, 2);
          const double corr = 3*delta * M2 / (i+1.0);
          return std::pow(delta, 3) * norm - corr;
        };
        const auto deltaM4 = [](double i, double delta, double M2, double M3) {
          const double norm = i * (i*i - i + 1.0) / std::pow(i+1.0, 3);
          const double cor1 = 6 * std::pow(delta, 2) * M2 / std::pow(i+1.0, 2);
          const double cor2 = 4 * delta * M3 / (i+1.0);
          return std::pow(delta, 4) * norm + cor1 - cor2;
        };

        printf("%s statistics %s.\n", name.c_str(), header.c_str());
        {
            double mean = 0, var = 0, skew = 0, kurt = 0;
            double min =  HUGE_VAL, max = -HUGE_VAL;

            for (unsigned int i = 0; i < N; ++i) {
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }

            double M2 = 0, M3 = 0, M4 = 0;
            for (unsigned int i = 0; i < N; ++i) {
                const double delta = data[i] - mean;
                const double dM2 = deltaM2(i, delta);
                const double dM3 = deltaM3(i, delta, M2);
                const double dM4 = deltaM4(i, delta, M2, M3);
                mean += delta / (i+1.0);
                M4 += dM4;
                M3 += dM3;
                M2 += dM2;
            }
            var  = M2 / (N - 1);
            skew = std::sqrt(N) * M3 / std::pow(M2 + 2e-16, 1.5);
            kurt = N * M4 / (M2 * M2 + 2e-16) - 3;
            printf("\tMesh spacing: mean=%e; std=%e; skew=%e; kurt=%e; min=%e; max=%e\n",
                   mean, std::sqrt(var), skew, kurt, min, max);
        }
        {
            double mean = 0, var = 0, skew = 0, kurt = 0;
            double min =  HUGE_VAL, max = -HUGE_VAL;

            for (unsigned int i = 1; i < N; ++i) {
                const double r = data[i]/data[i-1];
                if (r < min) min = r;
                if (r > max) max = r;
            }

            double M2 = 0, M3 = 0, M4 = 0;
            for (unsigned int i = 1; i < N; ++i) {
                const double r = data[i]/data[i-1];
                const double delta = r - mean;
                const double dM2 = deltaM2(i, delta);
                const double dM3 = deltaM3(i, delta, M2);
                const double dM4 = deltaM4(i, delta, M2, M3);
                mean += delta / (i+1.0);
                M4 += dM4;
                M3 += dM3;
                M2 += dM2;
            }
            var  = M2 / (N - 1);
            skew = std::sqrt(N) * M3 / std::pow(M2 + 2e-16, 1.5);
            kurt = N * M4 / (M2 * M2 + 2e-16) - 3;
            printf("\tGrowth factor: mean=%e; std=%e; skew=%e; kurt=%e; min=%e; max=%e\n",
                   mean, std::sqrt(var), skew, kurt, min, max);
        }
    }
};

CubismUP_3D_NAMESPACE_END
#endif // NonUniformScheme
