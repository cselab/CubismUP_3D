//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#ifndef CubismUP_3D_aligned_allocator_h
#define CubismUP_3D_aligned_allocator_h

#include <cstdlib>
#include <memory>

//#include <malloc.h>
// ALIGNMENT must be a power of 2 !
static constexpr int ALIGNMENT = 32;

CubismUP_3D_NAMESPACE_BEGIN

template <typename T>
class aligned_allocator {
  public:
    typedef T*              pointer;
    typedef T const*        const_pointer;
    typedef T&              reference;
    typedef T const&        const_reference;
    typedef T               value_type;
    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;

    template <typename U>
    struct rebind { typedef aligned_allocator<U> other; };

    aligned_allocator() noexcept { }

    aligned_allocator(aligned_allocator const& a) noexcept { }

    template <typename S>
    aligned_allocator(aligned_allocator<S> const& b) noexcept { }

    pointer allocate(size_type n)
    {
      pointer p;
      if(posix_memalign(reinterpret_cast<void**>(&p), ALIGNMENT, n*sizeof(T) ))
          throw std::bad_alloc();
      return p;
    }

    void deallocate(pointer p, size_type n) noexcept { std::free(p); }

    size_type max_size() const noexcept
    {
        std::allocator<T> a;
        return a.max_size();
    }

    template <typename C, class... Args>
    void construct(C* c, Args&&... args)
    {
        new ((void*)c) C(std::forward<Args>(args)...);
    }

    template <typename C>
    void destroy(C* c) { c->~C(); }

    bool operator == (aligned_allocator const & a2) const noexcept { return 1; }

    bool operator != (aligned_allocator const & a2) const noexcept { return 0; }

    template <typename S>
    bool operator == (aligned_allocator<S> const&b) const noexcept { return 0; }

    template <typename S>
    bool operator != (aligned_allocator<S> const&b) const noexcept { return 1; }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_aligned_allocator_h
