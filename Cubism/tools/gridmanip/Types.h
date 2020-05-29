// File       : Types.h
// Created    : Sat May 04 2019 09:41:16 PM (+0200)
// Author     : Fabian Wermelinger
// Description: Type declarations for grid and boundary processing
// Copyright 2019 ETH Zurich. All Rights Reserved.
#ifndef TYPES_H_AESHUI1N
#define TYPES_H_AESHUI1N

#ifdef _FLOAT_PRECISION_
typedef float Real;
typedef float DumpReal;
#else
typedef double Real;
typedef double DumpReal;
#endif

// scalar field pass-through streamer.  assumes that the element type TElement
// is a struct with field member .scalar.
struct StreamerScalar {
    static const std::string NAME;
    static const std::string EXT;
    static const int NCHANNELS = 1;
    static const int CLASS = 0;

    template <typename TBlock, typename TReal>
    static inline void operate(const TBlock &b,
                               const int ix,
                               const int iy,
                               const int iz,
                               TReal output[NCHANNELS])
    {
        typedef typename TBlock::ElementType TElement;
        const TElement &el = b(ix, iy, iz);
        output[0] = el.scalar;
    }

    template <typename TBlock, typename TReal>
    static inline void operate(TBlock &b,
                               const TReal input[NCHANNELS],
                               const int ix,
                               const int iy,
                               const int iz)
    {
        typedef typename TBlock::ElementType TElement;
        TElement &el = b(ix, iy, iz);
        el.scalar = input[0];
    }

    static const char *getAttributeName() { return "Scalar"; }
};

const std::string StreamerScalar::NAME = "Scalar";
const std::string StreamerScalar::EXT = "-scalar";

struct GridElement {
    using RealType = Real;

    RealType scalar;

    void clear() { scalar = 0.0; }

    GridElement &operator=(const GridElement &gp)
    {
        this->scalar = gp.scalar;
        return *this;
    }

    GridElement &operator+=(const GridElement &rhs)
    {
        scalar += rhs.scalar;
        return *this;
    }

    GridElement &operator-=(const GridElement &rhs)
    {
        scalar -= rhs.scalar;
        return *this;
    }

    GridElement &operator*=(const RealType c)
    {
        scalar *= c;
        return *this;
    }

    friend GridElement operator+(GridElement lhs, const GridElement &rhs)
    {
        return (lhs += rhs);
    }

    friend GridElement operator-(GridElement lhs, const GridElement &rhs)
    {
        return (lhs -= rhs);
    }

    friend GridElement operator*(GridElement lhs, const RealType c)
    {
        return (lhs *= c);
    }

    friend GridElement operator*(const RealType c, GridElement rhs)
    {
        return (rhs *= c);
    }
};

struct GridBlock {
    static const int sizeX = _BLOCKSIZE_;
    static const int sizeY = _BLOCKSIZE_;
    static const int sizeZ = _BLOCKSIZE_;

    static const int gptfloats =
        sizeof(GridElement) / sizeof(GridElement::RealType);

    using RealType = typename GridElement::RealType;
    using ElementType = GridElement;
    using element_type = GridElement;

    GridElement __attribute__((__aligned__(CUBISM_ALIGNMENT)))
    data[_BLOCKSIZE_][_BLOCKSIZE_][_BLOCKSIZE_];

    RealType __attribute__((__aligned__(CUBISM_ALIGNMENT)))
    tmp[_BLOCKSIZE_][_BLOCKSIZE_][_BLOCKSIZE_][gptfloats];

    void clear_data()
    {
        const int N = sizeX * sizeY * sizeZ;
        GridElement *const e = &data[0][0][0];
        for (int i = 0; i < N; ++i) {
            e[i].clear();
        }
    }

    void clear_tmp()
    {
        const int N = sizeX * sizeY * sizeZ * gptfloats;

        RealType *const e = &tmp[0][0][0][0];
        for (int i = 0; i < N; ++i) {
            e[i] = 0;
        }
    }

    void clear()
    {
        clear_data();
        clear_tmp();
    }

    inline GridElement &operator()(int ix, int iy = 0, int iz = 0)
    {
        assert(ix >= 0 && ix < sizeX);
        assert(iy >= 0 && iy < sizeY);
        assert(iz >= 0 && iz < sizeZ);
        return data[iz][iy][ix];
    }

    inline const GridElement &operator()(int ix, int iy = 0, int iz = 0) const
    {
        assert(ix >= 0 && ix < sizeX);
        assert(iy >= 0 && iy < sizeY);
        assert(iz >= 0 && iz < sizeZ);
        return data[iz][iy][ix];
    }
};

template <typename TBlock, template <typename> class Alloc = std::allocator>
class ExtrapolatingBoundaryTensorial : public cubism::BlockLab<TBlock, Alloc>
{
    using ElementTypeBlock = typename TBlock::ElementType;

public:
    std::string name() const override
    {
        return "ExtrapolatingBoundaryTensorial";
    }
    bool is_xperiodic() override { return false; }
    bool is_yperiodic() override { return false; }
    bool is_zperiodic() override { return false; }

    ExtrapolatingBoundaryTensorial() = default;

    void _apply_bc(const cubism::BlockInfo &info, const Real t = 0) override
    {
        if (info.index[0] == 0)
            this->template extrapolate_<0, 0>();
        if (info.index[0] == this->NX - 1)
            this->template extrapolate_<0, 1>();
        if (info.index[1] == 0)
            this->template extrapolate_<1, 0>();
        if (info.index[1] == this->NY - 1)
            this->template extrapolate_<1, 1>();
        if (info.index[2] == 0)
            this->template extrapolate_<2, 0>();
        if (info.index[2] == this->NZ - 1)
            this->template extrapolate_<2, 1>();
    }

private:
    int start_[3], end_[3];

    template <int dir, int side>
    void setupStartEnd_()
    {
        start_[0] = dir == 0
                        ? (side == 0 ? this->m_stencilStart[0] : TBlock::sizeX)
                        : 0;
        start_[1] = dir == 1
                        ? (side == 0 ? this->m_stencilStart[1] : TBlock::sizeY)
                        : 0;
        start_[2] = dir == 2
                        ? (side == 0 ? this->m_stencilStart[2] : TBlock::sizeZ)
                        : 0;

        end_[0] =
            dir == 0
                ? (side == 0 ? 0 : TBlock::sizeX + this->m_stencilEnd[0] - 1)
                : TBlock::sizeX;
        end_[1] =
            dir == 1
                ? (side == 0 ? 0 : TBlock::sizeY + this->m_stencilEnd[1] - 1)
                : TBlock::sizeY;
        end_[2] =
            dir == 2
                ? (side == 0 ? 0 : TBlock::sizeZ + this->m_stencilEnd[2] - 1)
                : TBlock::sizeZ;
    }

    ElementTypeBlock &accessCacheBlock_(int ix, int iy, int iz)
    {
        return this->m_cacheBlock->Access(ix - this->m_stencilStart[0],
                                          iy - this->m_stencilStart[1],
                                          iz - this->m_stencilStart[2]);
    }

    template <int dir, int side>
    void extrapolate_()
    {
        this->template setupStartEnd_<dir, side>();

        // faces
        for (int iz = start_[2]; iz < end_[2]; iz++)
            for (int iy = start_[1]; iy < end_[1]; iy++)
                for (int ix = start_[0]; ix < end_[0]; ix++) {
                    accessCacheBlock_(ix, iy, iz) = accessCacheBlock_(
                        dir == 0 ? (side == 0 ? 0 : TBlock::sizeX - 1) : ix,
                        dir == 1 ? (side == 0 ? 0 : TBlock::sizeY - 1) : iy,
                        dir == 2 ? (side == 0 ? 0 : TBlock::sizeZ - 1) : iz);
                }

        // edges and corners
        {
            int s_[3], e_[3];
            const int bsize[3] = {TBlock::sizeX, TBlock::sizeY, TBlock::sizeZ};

            s_[dir] =
                this->m_stencilStart[dir] * (1 - side) + bsize[dir] * side;
            e_[dir] = (bsize[dir] - 1 + this->m_stencilEnd[dir]) * side;

            const int d1 = (dir + 1) % 3;
            const int d2 = (dir + 2) % 3;

            for (int b = 0; b < 2; ++b)
                for (int a = 0; a < 2; ++a) {
                    s_[d1] = this->m_stencilStart[d1] +
                             a * b * (bsize[d1] - this->m_stencilStart[d1]);
                    s_[d2] =
                        this->m_stencilStart[d2] +
                        (a - a * b) * (bsize[d2] - this->m_stencilStart[d2]);

                    e_[d1] = (1 - b + a * b) *
                             (bsize[d1] - 1 + this->m_stencilEnd[d1]);
                    e_[d2] = (a + b - a * b) *
                             (bsize[d2] - 1 + this->m_stencilEnd[d2]);

                    for (int iz = s_[2]; iz < e_[2]; iz++)
                        for (int iy = s_[1]; iy < e_[1]; iy++)
                            for (int ix = s_[0]; ix < e_[0]; ix++) {
                                accessCacheBlock_(ix, iy, iz) =
                                    dir == 0
                                        ? accessCacheBlock_(
                                              side * (bsize[0] - 1), iy, iz)
                                        : (dir == 1
                                               ? accessCacheBlock_(
                                                     ix,
                                                     side * (bsize[1] - 1),
                                                     iz)
                                               : accessCacheBlock_(
                                                     ix,
                                                     iy,
                                                     side * (bsize[2] - 1)));
                            }
                }
        }
    }
};

enum class IOType { In = 0, Out };

template <typename TGrid, IOType Direction>
class InputOutputFactory
{
public:
    InputOutputFactory(const bool isroot, cubism::ArgumentParser &p)
        : m_isroot(isroot), m_parser(p), finput(nullptr), foutput(nullptr)
    {
        if (Direction == IOType::In) {
            setInputStreamer_();
        } else if (Direction == IOType::Out) {
            setOutputStreamer_();
        } else {
            if (m_isroot)
                std::cerr << "ERROR: Unknown Direction type" << std::endl;
            abort();
        }
    }

    using InStreamer = void (*)(TGrid &,
                                const std::string &,
                                const std::string &);
    using OutStreamer = void (*)(const TGrid &,
                                 const int,
                                 const Real,
                                 const std::string &,
                                 const std::string &,
                                 const bool);

    void operator()(TGrid &grid,
                    const std::string &fname,
                    const std::string &path = ".")
    {
        if (Direction == IOType::In) {
            finput(grid, fname, path);
        } else {
            foutput(grid, 0, 0.0, fname, path, true);
        }
    }

private:
    const bool m_isroot;
    cubism::ArgumentParser &m_parser;

    InStreamer finput;
    OutStreamer foutput;

    void setOutputStreamer_()
    {
        const std::string save_format =
            m_parser("save_format").asString("cubism_h5");

        if (save_format == "cubism_h5") {
            foutput = &cubism::DumpHDF5_MPI<StreamerScalar, DumpReal, TGrid>;
        } else {
            if (m_isroot)
                std::cerr << "ERROR: No suitable save format chosen"
                          << std::endl;
            abort();
        }

        if (foutput == nullptr) {
            if (m_isroot)
                std::cerr << "ERROR: NULL kernel" << std::endl;
            abort();
        }
    }

    void setInputStreamer_()
    {
        const std::string read_format =
            m_parser("read_format").asString("cubism_h5");

        if (read_format == "cubism_h5") {
            finput = &cubism::ReadHDF5_MPI<StreamerScalar, DumpReal, TGrid>;
        } else {
            if (m_isroot)
                std::cerr << "ERROR: No suitable input format chosen"
                          << std::endl;
            abort();
        }

        if (finput == nullptr) {
            if (m_isroot)
                std::cerr << "ERROR: NULL kernel" << std::endl;
            abort();
        }
    }
};

#endif /* TYPES_H_AESHUI1N */
