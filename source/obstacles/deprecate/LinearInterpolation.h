//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

#ifndef CubismUP_3D_LinearInterpolation_h
#define CubismUP_3D_LinearInterpolation_h

#include "operators/GenericOperator.h"
#include "operators/GenericCoordinator.h"
#include <vector>

CubismUP_3D_NAMESPACE_BEGIN

class LinearInterpolation;

template <typename Getter, typename Setter>
class LinearInterpolationKernel : public GenericLabOperator
{
    const LinearInterpolation &owner;
    Getter &getter;
    Setter &setter;
public:
    LinearInterpolationKernel(const LinearInterpolation &owner,
                              Getter &getter,
                              Setter &setter,
                              const std::vector<int> &components)
            : owner(owner),
              getter(getter),
              setter(setter)
    {
        stencil = StencilInfo(-1, -1, -1, 2, 2, 2, true, components);
    };

    template <typename Lab, typename BlockType>
    void operator()(Lab &lab, const BlockInfo &info, BlockType &o) const;
};


class LinearInterpolation : public GenericCoordinator
{
public:
    LinearInterpolation(SimulationData&s)
            : GenericCoordinator(s) { }

    /*
     * Mesh to particle linear interpolation algorithm.
     *
     * For each given point, interpolate the value of the field.
     *
     * Arguments:
     *   - points - Array of the points. (*)
     *   - getter - Lambda of a single argument (BlockLab), returning the value
     *              to be interpolated (**).
     *   - setter - Lambda of two arguments (point ID, interpolated value).
     *   - components - Stencil components.
     *
     * (*) Points should support operator [] for accessing x, y, z coordinates.
     * (**) Returned value type X must support operators <double> * X and X + X.
     */
    template <typename Array, typename Getter, typename Setter>
    void interpolate(const Array &points,
                     Getter getter,
                     Setter setter,
                     const std::vector<int> &components);

    // GenericCoordinator stuff we don't care about now.
    virtual ~LinearInterpolation() { }
    void operator()(const double /* dt */) override { abort(); }
    std::string getName(void) { return "LinearInterpolation"; }

    struct Particle {
        int id;
        double pos[3];
    };

    int N[3];
    std::vector<std::vector<Particle>> particles;
};


// Implementation.
template <typename Getter, typename Setter>
template <typename Lab, typename BlockType>
void LinearInterpolationKernel<Getter, Setter>::operator()(
        Lab &lab,
        const BlockInfo &info,
        BlockType &o) const
{
    typedef typename FluidGridMPI::BlockType Block;
    const int block_index = info.index[0] + owner.N[0] * (
                            info.index[1] + owner.N[1] * info.index[2]);
    const double invh = 1.0 / info.h_gridpoint;

    for (const auto &part : owner.particles[block_index]) {
        // FIXME: This calculation of the index is not 100% consistent with the
        // other one.
        // Get position in index space within block.
        const double ipos[3] = {
            invh * (part.pos[0] - info.origin[0]),
            invh * (part.pos[1] - info.origin[1]),
            invh * (part.pos[2] - info.origin[2]),
        };
        const int idx[3] = {
            std::max(0, std::min((int)ipos[0], Block::sizeArray[0] - 1)),
            std::max(0, std::min((int)ipos[1], Block::sizeArray[1] - 1)),
            std::max(0, std::min((int)ipos[2], Block::sizeArray[2] - 1)),
        };

        // Compute 1D weights.
        const double w[3] = {
            ipos[0] - idx[0],
            ipos[1] - idx[1],
            ipos[2] - idx[2],
        };

        // Do M2P interpolation.
        const double w000 = (1 - w[0]) * (1 - w[1]) * (1 - w[2]);
        const double w010 = (1 - w[0]) * (    w[1]) * (1 - w[2]);
        const double w100 = (    w[0]) * (1 - w[1]) * (1 - w[2]);
        const double w110 = (    w[0]) * (    w[1]) * (1 - w[2]);
        const double w001 = (1 - w[0]) * (1 - w[1]) * (    w[2]);
        const double w011 = (1 - w[0]) * (    w[1]) * (    w[2]);
        const double w101 = (    w[0]) * (1 - w[1]) * (    w[2]);
        const double w111 = (    w[0]) * (    w[1]) * (    w[2]);
        setter(part.id,
               w000 * getter(lab.read(idx[0]    , idx[1]    , idx[2]    ))
             + w010 * getter(lab.read(idx[0]    , idx[1] + 1, idx[2]    ))
             + w100 * getter(lab.read(idx[0] + 1, idx[1]    , idx[2]    ))
             + w110 * getter(lab.read(idx[0] + 1, idx[1] + 1, idx[2]    ))
             + w001 * getter(lab.read(idx[0]    , idx[1]    , idx[2] + 1))
             + w011 * getter(lab.read(idx[0]    , idx[1] + 1, idx[2] + 1))
             + w101 * getter(lab.read(idx[0] + 1, idx[1]    , idx[2] + 1))
             + w111 * getter(lab.read(idx[0] + 1, idx[1] + 1, idx[2] + 1)));
    }
}


template <typename Array, typename Getter, typename Setter>
void LinearInterpolation::interpolate(
        const Array &points,
        Getter getter,
        Setter setter,
        const std::vector<int> &components)
{
    N[0] = grid->getBlocksPerDimension(0);
    N[1] = grid->getBlocksPerDimension(1);
    N[2] = grid->getBlocksPerDimension(2);

    particles.clear();
    particles.resize(N[0] * N[1] * N[2]);

    // Map particles to CUBISM domain [0, 1] and put them in different blocks.
    for (decltype(points.size()) i = 0; i < points.size(); ++i) {
        const auto &point = points[i];
        Particle part;
        part.id = i;
        part.pos[0] = point[0];
        part.pos[1] = point[1];
        part.pos[2] = point[2];

        // Find block.
        const int index[3] = {
            std::max(0, std::min((int)(part.pos[0] * N[0]), N[0] - 1)),
            std::max(0, std::min((int)(part.pos[1] * N[1]), N[1] - 1)),
            std::max(0, std::min((int)(part.pos[2] * N[2]), N[2] - 1)),
        };
        const int idx = index[0] + N[0] * (index[1] + N[1] * index[2]);
        particles[idx].push_back(part);
    }

    LinearInterpolationKernel<decltype(getter), decltype(setter)>
            kernel(*this, getter, setter, components);

    compute(kernel);
}

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_LinearInterpolation_h
