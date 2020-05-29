//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include <array>
#include <limits>

CubismUP_3D_NAMESPACE_BEGIN

using SymM = std::array<double,6>;
using GenM = std::array<double,9>;
using GenV = std::array<double,3>;

static inline SymM invertSym(const SymM J)
{
  const double detJ = J[0] * (J[1] * J[2] - J[5] * J[5])+
                      J[3] * (J[4] * J[5] - J[2] * J[3])+
                      J[4] * (J[3] * J[5] - J[1] * J[4]);
  if( std::fabs(detJ) <= std::numeric_limits<double>::min() ) // really zero
  {
    return SymM {{ 0, 0, 0, 0, 0, 0 }};
  }
  else
  {
    return SymM {{
     (J[1] * J[2] - J[5] * J[5]) / detJ,
     (J[0] * J[2] - J[4] * J[4]) / detJ,
     (J[0] * J[1] - J[3] * J[3]) / detJ,
     (J[4] * J[5] - J[2] * J[3]) / detJ,
     (J[3] * J[5] - J[1] * J[4]) / detJ,
     (J[3] * J[4] - J[0] * J[5]) / detJ
    }};
  }
}

static inline GenV multSymVec(const SymM J, const GenV V)
{
  return GenV {{
    J[0] * V[0] + J[3] * V[1] + J[4] * V[2],
    J[3] * V[0] + J[1] * V[1] + J[5] * V[2],
    J[4] * V[0] + J[5] * V[1] + J[2] * V[2]
 }};
}

static inline GenV multGenVec(const GenM J, const GenV V)
{
  return GenV {{
    J[0] * V[0] + J[1] * V[1] + J[2] * V[2],
    J[3] * V[0] + J[4] * V[1] + J[5] * V[2],
    J[6] * V[0] + J[7] * V[1] + J[8] * V[2]
 }};
}

static inline GenM multSyms(const SymM J, const SymM G)
{
  return GenM {{
    G[0]*J[0] + G[3]*J[3] + G[4]*J[4],
    G[0]*J[3] + G[3]*J[1] + G[4]*J[5],
    G[0]*J[4] + G[4]*J[2] + G[3]*J[5],
    G[3]*J[0] + G[1]*J[3] + G[5]*J[4],
    G[1]*J[1] + G[3]*J[3] + G[5]*J[5],
    G[1]*J[5] + G[3]*J[4] + G[5]*J[2],
    G[4]*J[0] + G[2]*J[4] + G[5]*J[3],
    G[5]*J[1] + G[2]*J[5] + G[4]*J[3],
    G[2]*J[2] + G[4]*J[4] + G[5]*J[5]
  }};
}

static inline GenM invertGen(const GenM S)
{
  const double detS =  S[0]*S[4]*S[8] - S[0]*S[5]*S[7]
                     + S[1]*S[5]*S[6] - S[1]*S[3]*S[8]
                     + S[2]*S[3]*S[7] - S[2]*S[4]*S[6];
  if( std::fabs(detS) <= std::numeric_limits<double>::min() ) // really zero
  {
    return GenM {{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }};
  }
  else
  {
    return GenM {{
        ( S[4]*S[8] - S[5]*S[7] ) / detS,
      - ( S[1]*S[8] - S[2]*S[7] ) / detS,
        ( S[1]*S[5] - S[2]*S[4] ) / detS,
      - ( S[3]*S[8] - S[5]*S[6] ) / detS,
        ( S[0]*S[8] - S[2]*S[6] ) / detS,
      - ( S[0]*S[5] - S[2]*S[3] ) / detS,
        ( S[3]*S[7] - S[4]*S[6] ) / detS,
      - ( S[0]*S[7] - S[1]*S[6] ) / detS,
        ( S[0]*S[4] - S[1]*S[3] ) / detS
    }};
  }
}

static inline GenM multGens(const GenM S, const GenM G)
{
  return GenM {{
    G[0]*S[0] + G[3]*S[1] + G[6]*S[2],
    G[1]*S[0] + G[4]*S[1] + G[7]*S[2],
    G[2]*S[0] + G[5]*S[1] + G[8]*S[2],
    G[0]*S[3] + G[3]*S[4] + G[6]*S[5],
    G[1]*S[3] + G[4]*S[4] + G[7]*S[5],
    G[2]*S[3] + G[5]*S[4] + G[8]*S[5],
    G[0]*S[6] + G[3]*S[7] + G[6]*S[8],
    G[1]*S[6] + G[4]*S[7] + G[7]*S[8],
    G[2]*S[6] + G[5]*S[7] + G[8]*S[8]
  }};
}
CubismUP_3D_NAMESPACE_END
