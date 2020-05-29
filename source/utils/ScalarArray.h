//
//  Cubism3D
//  Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

#ifndef CubismUP_3D_utils_ScalarArray_h
#define CubismUP_3D_utils_ScalarArray_h

#include "../Base.h"

#include <array>

CubismUP_3D_NAMESPACE_BEGIN

/*
 * Extension of std::array that implements the operators +, - between
 * themselves and the operator * with a scalar.
 *
 * Used for interpolation API.
 */
template <typename T, size_t N>
struct ScalarArray {
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using differencee_type = ptrdiff_t;

  value_type v[N];

  reference operator[](const size_type k) {
    return v[k];
  }
  const_reference operator[](const size_type k) const {
    return v[k];
  }
  size_type size() const {
    return N;
  }
  pointer data() {
    return v;
  }
  const_pointer data() const {
    return v;
  }
  iterator begin() {
    return iterator(data());
  }
  const_iterator begin() const {
    return const_iterator(data());
  }
  const_iterator cbegin() const {
    return const_iterator(data());
  }
  iterator end() {
    return iterator(data() + N);
  }
  const_iterator end() const {
    return const_iterator(data() + N);
  }
  const_iterator cend() const {
    return const_iterator(data());
  }

  // This is not the optimal implementation (values get constructed and then
  // replaced), but compiler should to the job of optimizing unnecessary
  // parts away.
  friend inline ScalarArray operator+(const ScalarArray &A, const ScalarArray &B) {
    ScalarArray<T, N> result;
    for (size_type i = 0; i < N; ++i)
      result[i] = A[i] + B[i];
    return result;
  }

  friend inline ScalarArray operator-(const ScalarArray &A, const ScalarArray &B) {
    ScalarArray<T, N> result;
    for (size_type i = 0; i < N; ++i)
      result[i] = A[i] - B[i];
    return result;
  }

  friend inline ScalarArray operator*(const T &A, const ScalarArray &B) {
    ScalarArray<T, N> result;
    for (size_type i = 0; i < N; ++i)
      result[i] = A * B[i];
    return result;
  }

  friend inline ScalarArray operator*(const ScalarArray &A, const T &B) {
    ScalarArray<T, N> result;
    for (size_type i = 0; i < N; ++i)
      result[i] = A[i] * B;
    return result;
  }
};

CubismUP_3D_NAMESPACE_END

#endif // CubismUP_3D_utils_ScalarArray_h
