#pragma once

#include "for_each.hpp"
#include "reduce.hpp"

/**
@file taskflow/hip/algorithm/find.hpp
@brief hip find algorithms include file
*/

namespace tf::detail {

/** @private */
template <typename T>
struct hipFindPair {

  T key;
  unsigned index;

  __device__ operator unsigned () const { return index; }
};

/** @private */
template <typename P, typename I, typename U>
void hip_find_if_loop(P&& p, I input, unsigned count, unsigned* idx, U pred) {

  if(count == 0) {
    hip_single_task(p, [=] __device__ () { *idx = 0; });
    return;
  }

  using E = std::decay_t<P>;

  auto B = (count + E::nv - 1) / E::nv;

  // set the index to the maximum
  hip_single_task(p, [=] __device__ () { *idx = count; });

  // launch the kernel to atomic-find the minimum
  hip_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {

    __shared__ unsigned shm_id;

    if(!tid) {
      shm_id = count;
    }

    __syncthreads();

    auto tile = hip_get_tile(bid, E::nv, count);

    auto x = hip_mem_to_reg_strided<E::nt, E::vt>(
      input + tile.begin, tid, tile.count()
    );

    auto id = count;

    for(unsigned i=0; i<E::vt; i++) {
      auto j = E::nt*i + tid;
      if(j < tile.count() && pred(x[i])) {
        id = j + tile.begin;
        break;
      }
    }

    // Note: the reduce version is not faster though
    // reduce to a scalar per block.
    //__shared__ typename hipBlockReduce<E::nt, unsigned>::Storage shm;

    //id = hipBlockReduce<E::nt, unsigned>()(
    //  tid,
    //  id,
    //  shm,
    //  (tile.count() < E::nt ? tile.count() : E::nt),
    //  hip_minimum<unsigned>{},
    //  false
    //);

    // only need the minimum id
    atomicMin(&shm_id, id);
    __syncthreads();

    // reduce all to the global memory
    if(!tid) {
      atomicMin(idx, shm_id);
      //atomicMin(idx, id);
    }
  });
}

/** @private */
template <typename P, typename I, typename O>
void hip_min_element_loop(
  P&& p, I input, unsigned count, unsigned* idx, O op, void* ptr
) {

  if(count == 0) {
    hip_single_task(p, [=] __device__ () { *idx = 0; });
    return;
  }

  using T = hipFindPair<typename std::iterator_traits<I>::value_type>;

  hip_uninitialized_reduce_loop(p,
    hip_make_load_iterator<T>([=]__device__(auto i){
      return T{*(input+i), i};
    }),
    count,
    idx,
    [=] __device__ (const auto& a, const auto& b) {
      return op(a.key, b.key) ? a : b;
    },
    ptr
  );
}

/** @private */
template <typename P, typename I, typename O>
void hip_max_element_loop(
  P&& p, I input, unsigned count, unsigned* idx, O op, void* ptr
) {

  if(count == 0) {
    hip_single_task(p, [=] __device__ () { *idx = 0; });
    return;
  }

  using T = hipFindPair<typename std::iterator_traits<I>::value_type>;

  hip_uninitialized_reduce_loop(p,
    hip_make_load_iterator<T>([=]__device__(auto i){
      return T{*(input+i), i};
    }),
    count,
    idx,
    [=] __device__ (const auto& a, const auto& b) {
      return op(a.key, b.key) ? b : a;
    },
    ptr
  );
}

}  // end of namespace tf::detail ---------------------------------------------

namespace tf {


// ----------------------------------------------------------------------------
// hip_find_if
// ----------------------------------------------------------------------------

/**
@brief finds the index of the first element that satisfies the given criteria

@tparam P execution policy type
@tparam I input iterator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param idx pointer to the index of the found element
@param op unary operator which returns @c true for the required element

The function launches kernels asynchronously to find the index @c idx of the
first element in the range <tt>[first, last)</tt>
such that <tt>op(*(first+idx))</tt> is true.
This is equivalent to the parallel execution of the following loop:

@code{.cpp}
unsigned idx = 0;
for(; first != last; ++first, ++idx) {
  if (p(*first)) {
    return idx;
  }
}
return idx;
@endcode
*/
template <typename P, typename I, typename U>
void hip_find_if(
  P&& p, I first, I last, unsigned* idx, U op
) {
  detail::hip_find_if_loop(p, first, std::distance(first, last), idx, op);
}

// ----------------------------------------------------------------------------
// hipFlow
// ----------------------------------------------------------------------------

// Function: find_if
template <typename I, typename U>
hipTask hipFlow::find_if(I first, I last, unsigned* idx, U op) {
  return capture([=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.find_if(first, last, idx, op);
  });
}

// Function: find_if
template <typename I, typename U>
void hipFlow::find_if(hipTask task, I first, I last, unsigned* idx, U op) {
  capture(task, [=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.find_if(first, last, idx, op);
  });
}

// ----------------------------------------------------------------------------
// hipFlowCapturer
// ----------------------------------------------------------------------------

// Function: find_if
template <typename I, typename U>
hipTask hipFlowCapturer::find_if(I first, I last, unsigned* idx, U op) {
  return on([=](hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_find_if(p, first, last, idx, op);
  });
}

// Function: find_if
template <typename I, typename U>
void hipFlowCapturer::find_if(
  hipTask task, I first, I last, unsigned* idx, U op
) {
  on(task, [=](hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_find_if(p, first, last, idx, op);
  });
}

// ----------------------------------------------------------------------------
// hip_min_element
// ----------------------------------------------------------------------------

/**
@brief queries the buffer size in bytes needed to call tf::hip_min_element

@tparam P execution policy type
@tparam T value type

@param count number of elements to search

The function is used to decide the buffer size in bytes for calling
tf::hip_min_element.
*/
template <typename P, typename T>
unsigned hip_min_element_buffer_size(unsigned count) {
  return hip_reduce_buffer_size<P, detail::hipFindPair<T>>(count);
}

/**
@brief finds the index of the minimum element in a range

@tparam P execution policy type
@tparam I input iterator type
@tparam O comparator type

@param p execution policy object
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param idx solution index of the minimum element
@param op comparison function object
@param buf pointer to the buffer

The function launches kernels asynchronously to find
the smallest element in the range <tt>[first, last)</tt>
using the given comparator @c op.
You need to provide a buffer that holds at least
tf::hip_min_element_buffer_size bytes for internal use.
The function is equivalent to a parallel execution of the following loop:

@code{.cpp}
if(first == last) {
  return 0;
}
auto smallest = first;
for (++first; first != last; ++first) {
  if (op(*first, *smallest)) {
    smallest = first;
  }
}
return std::distance(first, smallest);
@endcode
*/
template <typename P, typename I, typename O>
void hip_min_element(P&& p, I first, I last, unsigned* idx, O op, void* buf) {
  detail::hip_min_element_loop(
    p, first, std::distance(first, last), idx, op, buf
  );
}

// ----------------------------------------------------------------------------
// hipFlowCapturer::min_element
// ----------------------------------------------------------------------------

// Function: min_element
template <typename I, typename O>
hipTask hipFlowCapturer::min_element(I first, I last, unsigned* idx, O op) {

  using T = typename std::iterator_traits<I>::value_type;

  auto bufsz = hip_min_element_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_min_element(p, first, last, idx, op, buf.get().data());
  });
}

// Function: min_element
template <typename I, typename O>
void hipFlowCapturer::min_element(
  hipTask task, I first, I last, unsigned* idx, O op
) {

  using T = typename std::iterator_traits<I>::value_type;

  auto bufsz = hip_min_element_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_min_element(p, first, last, idx, op, buf.get().data());
  });
}

// ----------------------------------------------------------------------------
// hipFlow::min_element
// ----------------------------------------------------------------------------

// Function: min_element
template <typename I, typename O>
hipTask hipFlow::min_element(I first, I last, unsigned* idx, O op) {
  return capture([=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.min_element(first, last, idx, op);
  });
}

// Function: min_element
template <typename I, typename O>
void hipFlow::min_element(
  hipTask task, I first, I last, unsigned* idx, O op
) {
  capture(task, [=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.min_element(first, last, idx, op);
  });
}

// ----------------------------------------------------------------------------
// hip_max_element
// ----------------------------------------------------------------------------

/**
@brief queries the buffer size in bytes needed to call tf::hip_max_element

@tparam P execution policy type
@tparam T value type

@param count number of elements to search

The function is used to decide the buffer size in bytes for calling
tf::hip_max_element.
*/
template <typename P, typename T>
unsigned hip_max_element_buffer_size(unsigned count) {
  return hip_reduce_buffer_size<P, detail::hipFindPair<T>>(count);
}

/**
@brief finds the index of the maximum element in a range

@tparam P execution policy type
@tparam I input iterator type
@tparam O comparator type

@param p execution policy object
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param idx solution index of the maximum element
@param op comparison function object
@param buf pointer to the buffer

The function launches kernels asynchronously to find
the largest element in the range <tt>[first, last)</tt>
using the given comparator @c op.
You need to provide a buffer that holds at least
tf::hip_max_element_buffer_size bytes for internal use.
The function is equivalent to a parallel execution of the following loop:

@code{.cpp}
if(first == last) {
  return 0;
}
auto largest = first;
for (++first; first != last; ++first) {
  if (op(*largest, *first)) {
    largest = first;
  }
}
return std::distance(first, largest);
@endcode
*/
template <typename P, typename I, typename O>
void hip_max_element(P&& p, I first, I last, unsigned* idx, O op, void* buf) {
  detail::hip_max_element_loop(
    p, first, std::distance(first, last), idx, op, buf
  );
}

// ----------------------------------------------------------------------------
// hipFlowCapturer::max_element
// ----------------------------------------------------------------------------

// Function: max_element
template <typename I, typename O>
hipTask hipFlowCapturer::max_element(I first, I last, unsigned* idx, O op) {

  using T = typename std::iterator_traits<I>::value_type;

  auto bufsz = hip_max_element_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_max_element(p, first, last, idx, op, buf.get().data());
  });
}

// Function: max_element
template <typename I, typename O>
void hipFlowCapturer::max_element(
  hipTask task, I first, I last, unsigned* idx, O op
) {

  using T = typename std::iterator_traits<I>::value_type;

  auto bufsz = hip_max_element_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_max_element(p, first, last, idx, op, buf.get().data());
  });
}

// ----------------------------------------------------------------------------
// hipFlow::max_element
// ----------------------------------------------------------------------------

// Function: max_element
template <typename I, typename O>
hipTask hipFlow::max_element(I first, I last, unsigned* idx, O op) {
  return capture([=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.max_element(first, last, idx, op);
  });
}

// Function: max_element
template <typename I, typename O>
void hipFlow::max_element(
  hipTask task, I first, I last, unsigned* idx, O op
) {
  capture(task, [=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.max_element(first, last, idx, op);
  });
}

}  // end of namespace tf -----------------------------------------------------


