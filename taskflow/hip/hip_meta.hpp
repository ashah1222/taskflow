#pragma once

#include "hip_execution_policy.hpp"

namespace tf {

// default warp size
inline constexpr unsigned hip_WARP_SIZE = 32;

// empty type
struct hipEmpty { };

// ----------------------------------------------------------------------------
// iterator unrolling
// ----------------------------------------------------------------------------

// Template unrolled looping construct.
template<unsigned i, unsigned count, bool valid = (i < count)>
struct hipIterate {
  template<typename F>
  __device__ static void eval(F f) {
    f(i);
    hipIterate<i + 1, count>::eval(f);
  }
};

template<unsigned i, unsigned count>
struct hipIterate<i, count, false> {
  template<typename F>
  __device__ static void eval(F) { }
};

template<unsigned begin, unsigned end, typename F>
__device__ void hip_iterate(F f) {
  hipIterate<begin, end>::eval(f);
}

template<unsigned count, typename F>
__device__ void hip_iterate(F f) {
  hip_iterate<0, count>(f);
}

template<unsigned count, typename T>
__device__ T reduce(const T(&x)[count]) {
  T y;
  hip_iterate<count>([&](auto i) { y = i ? x[i] + y : x[i]; });
  return y;
}

template<unsigned count, typename T>
__device__ void fill(T(&x)[count], T val) {
  hip_iterate<count>([&](auto i) { x[i] = val; });
}

// Invoke unconditionally.
template<unsigned nt, unsigned vt, typename F>
__device__ void hip_strided_iterate(F f, unsigned tid) {
  hip_iterate<vt>([=](auto i) { f(i, nt * i + tid); });
}

// Check range.
template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename F>
__device__ void hip_strided_iterate(F f, unsigned tid, unsigned count) {
  // Unroll the first vt0 elements of each thread.
  if(vt0 > 1 && count >= nt * vt0) {
    hip_strided_iterate<nt, vt0>(f, tid);    // No checking
  } else {
    hip_iterate<vt0>([=](auto i) {
      auto j = nt * i + tid;
      if(j < count) f(i, j);
    });
  }

  // TODO: seems dummy when vt0 == vt
  hip_iterate<vt0, vt>([=](auto i) {
    auto j = nt * i + tid;
    if(j < count) f(i, j);
  });
}

template<unsigned vt, typename F>
__device__ void hip_thread_iterate(F f, unsigned tid) {
  hip_iterate<vt>([=](auto i) { f(i, vt * tid + i); });
}

// ----------------------------------------------------------------------------
// hipRange
// ----------------------------------------------------------------------------

// hipRange
struct hipRange {
  unsigned begin, end;
  __device__ unsigned size() const { return end - begin; }
  __device__ unsigned count() const { return size(); }
  __device__ bool valid() const { return end > begin; }
};

inline __device__ hipRange hip_get_tile(unsigned b, unsigned nv, unsigned count) {
  return hipRange { nv * b, min(count, nv * (b + 1)) };
}


// ----------------------------------------------------------------------------
// hipArray
// ----------------------------------------------------------------------------

template<typename T, unsigned size>
struct hipArray {
  T data[size];

  __device__ T operator[](unsigned i) const { return data[i]; }
  __device__ T& operator[](unsigned i) { return data[i]; }

  hipArray() = default;
  hipArray(const hipArray&) = default;
  hipArray& operator=(const hipArray&) = default;

  // Fill the array with x.
  __device__ hipArray(T x) {
    hip_iterate<size>([&](unsigned i) { data[i] = x; });
  }
};

template<typename T>
struct hipArray<T, 0> {
  __device__ T operator[](unsigned) const { return T(); }
  __device__ T& operator[](unsigned) { return *(T*)nullptr; }
};

template<typename T, typename V, unsigned size>
struct hipKVArray {
  hipArray<T, size> keys;
  hipArray<V, size> vals;
};

// ----------------------------------------------------------------------------
// thread reg <-> global mem
// ----------------------------------------------------------------------------

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename I>
__device__ auto hip_mem_to_reg_strided(I mem, unsigned tid, unsigned count) {
  using T = typename std::iterator_traits<I>::value_type;
  hipArray<T, vt> x;
  hip_strided_iterate<nt, vt, vt0>(
    [&](auto i, auto j) { x[i] = mem[j]; }, tid, count
  );
  return x;
}

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t>
__device__ void hip_reg_to_mem_strided(
  hipArray<T, vt> x, unsigned tid, unsigned count, it_t mem) {

  hip_strided_iterate<nt, vt, vt0>(
    [=](auto i, auto j) { mem[j] = x[i]; }, tid, count
  );
}

template<unsigned nt, unsigned vt, unsigned vt0 = vt, typename I, typename O>
__device__ auto hip_transform_mem_to_reg_strided(
  I mem, unsigned tid, unsigned count, O op
) {
  using T = std::invoke_result_t<O, typename std::iterator_traits<I>::value_type>;
  hipArray<T, vt> x;
  hip_strided_iterate<nt, vt, vt0>(
    [&](auto i, auto j) { x[i] = op(mem[j]); }, tid, count
  );
  return x;
}

// ----------------------------------------------------------------------------
// thread reg <-> shared
// ----------------------------------------------------------------------------

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ void hip_reg_to_shared_thread(
  hipArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_thread must have at least nt * vt storage");

  hip_thread_iterate<vt>([&](auto i, auto j) { shared[j] = x[i]; }, tid);

  if(sync) __syncthreads();
}

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ auto hip_shared_to_reg_thread(
  const T (&shared)[shared_size], unsigned tid, bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_thread must have at least nt * vt storage");

  hipArray<T, vt> x;
  hip_thread_iterate<vt>([&](auto i, auto j) {
    x[i] = shared[j];
  }, tid);

  if(sync) __syncthreads();

  return x;
}

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ void hip_reg_to_shared_strided(
  hipArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "reg_to_shared_strided must have at least nt * vt storage");

  hip_strided_iterate<nt, vt>(
    [&](auto i, auto j) { shared[j] = x[i]; }, tid
  );

  if(sync) __syncthreads();
}

template<unsigned nt, unsigned vt, typename T, unsigned shared_size>
__device__ auto hip_shared_to_reg_strided(
  const T (&shared)[shared_size], unsigned tid, bool sync = true
) {

  static_assert(shared_size >= nt * vt,
    "shared_to_reg_strided must have at least nt * vt storage");

  hipArray<T, vt> x;
  hip_strided_iterate<nt, vt>([&](auto i, auto j) { x[i] = shared[j]; }, tid);
  if(sync) __syncthreads();

  return x;
}

template<
  unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t,
  unsigned shared_size
>
__device__ auto hip_reg_to_mem_thread(
  hipArray<T, vt> x, unsigned tid,
  unsigned count, it_t mem, T (&shared)[shared_size]
) {
  hip_reg_to_shared_thread<nt>(x, tid, shared);
  auto y = hip_shared_to_reg_strided<nt, vt>(shared, tid);
  hip_reg_to_mem_strided<nt, vt, vt0>(y, tid, count, mem);
}

template<
  unsigned nt, unsigned vt, unsigned vt0 = vt, typename T, typename it_t,
  unsigned shared_size
>
__device__ auto hip_mem_to_reg_thread(
  it_t mem, unsigned tid, unsigned count, T (&shared)[shared_size]
) {

  auto x = hip_mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);
  hip_reg_to_shared_strided<nt, vt>(x, tid, shared);
  auto y = hip_shared_to_reg_thread<nt, vt>(shared, tid);
  return y;
}

template<unsigned nt, unsigned vt, typename T, unsigned S>
__device__ auto hip_shared_gather(
  const T(&data)[S], hipArray<unsigned, vt> indices, bool sync = true
) {

  static_assert(S >= nt * vt,
    "shared_gather must have at least nt * vt storage");

  hipArray<T, vt> x;
  hip_iterate<vt>([&](auto i) { x[i] = data[indices[i]]; });

  if(sync) __syncthreads();

  return x;
}



// ----------------------------------------------------------------------------
// reg<->reg
// ----------------------------------------------------------------------------

template<unsigned nt, unsigned vt, typename T, unsigned S>
__device__ auto hip_reg_thread_to_strided(
  hipArray<T, vt> x, unsigned tid, T (&shared)[S]
) {
  hip_reg_to_shared_thread<nt>(x, tid, shared);
  return hip_shared_to_reg_strided<nt, vt>(shared, tid);
}

template<unsigned nt, unsigned vt, typename T, unsigned S>
__device__ auto hip_reg_strided_to_thread(
  hipArray<T, vt> x, unsigned tid, T (&shared)[S]
) {
  hip_reg_to_shared_strided<nt>(x, tid, shared);
  return hip_shared_to_reg_thread<nt, vt>(shared, tid);
}

// ----------------------------------------------------------------------------
// hipLoadStoreIterator
// ----------------------------------------------------------------------------

template<typename L, typename S, typename T, typename I>
struct hipLoadStoreIterator : std::iterator_traits<const T*> {

  L load;
  S store;
  I base;

  hipLoadStoreIterator(L load_, S store_, I base_) :
    load(load_), store(store_), base(base_) { }

  struct assign_t {
    L load;
    S store;
    I index;

    __device__ assign_t& operator=(T rhs) {
      static_assert(!std::is_same<S, hipEmpty>::value,
        "load_iterator is being stored to.");
      store(rhs, index);
      return *this;
    }
    __device__ operator T() const {
      static_assert(!std::is_same<L, hipEmpty>::value,
        "store_iterator is being loaded from.");
      return load(index);
    }
  };

  __device__ assign_t operator[](I index) const {
    return assign_t { load, store, base + index };
  }
  __device__ assign_t operator*() const {
    return assign_t { load, store, base };
  }

  __device__ hipLoadStoreIterator operator+(I offset) const {
    hipLoadStoreIterator cp = *this;
    cp += offset;
    return cp;
  }

  __device__ hipLoadStoreIterator& operator+=(I offset) {
    base += offset;
    return *this;
  }

  __device__ hipLoadStoreIterator operator-(I offset) const {
    hipLoadStoreIterator cp = *this;
    cp -= offset;
    return cp;
  }

  __device__ hipLoadStoreIterator& operator-=(I offset) {
    base -= offset;
    return *this;
  }
};

//template<typename T>
//struct trivial_load_functor {
//  template<typename I>
//  __device__ T operator()(I index) const {
//    return T();
//  }
//};

//template<typename T>
//struct trivial_store_functor {
//  template<typename I>
//  __device__ void operator()(T v, I index) const { }
//};

template <typename T, typename I = unsigned, typename L, typename S>
auto hip_make_load_store_iterator(L load, S store, I base = 0) {
  return hipLoadStoreIterator<L, S, T, I>(load, store, base);
}

template <typename T, typename I = unsigned, typename L>
auto hip_make_load_iterator(L load, I base = 0) {
  return hip_make_load_store_iterator<T>(load, hipEmpty(), base);
}

template <typename T, typename I = unsigned, typename S>
auto hip_make_store_iterator(S store, I base = 0) {
  return hip_make_load_store_iterator<T>(hipEmpty(), store, base);
}

// ----------------------------------------------------------------------------
// swap
// ----------------------------------------------------------------------------

template<typename T>
__device__ void hip_swap(T& a, T& b) {
  auto c = a;
  a = b;
  b = c;
}

// ----------------------------------------------------------------------------
// launch kernel
// ----------------------------------------------------------------------------

template<typename F, typename... args_t>
__global__ void hip_kernel(F f, args_t... args) {
  f(threadIdx.x, blockIdx.x, args...);
}

// ----------------------------------------------------------------------------
// operators
// ----------------------------------------------------------------------------

template <class T>
struct hip_plus{
  __device__ T operator()(T a, T b) const { return a + b; }
};

 template <class T>
struct hip_minus{
  __device__ T operator()(T a, T b) const { return a - b; }
};

template <class T>
struct hip_multiplies{
  __device__ T operator()(T a, T b) const { return a * b; }
};

template <class T>
struct hip_maximum{
  __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};

template <class T>
struct hip_minimum{
  __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};

template <class T>
struct hip_less{
  __device__ T operator()(T a, T b) const { return a < b; }
};

template <class T>
struct hip_greater{
  __device__ T operator()(T a, T b) const { return a > b; }
};

}  // end of namespace tf -----------------------------------------------------
