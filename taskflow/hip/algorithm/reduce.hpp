#pragma once

#include "../hipflow.hpp"

/**
@file taskflow/hip/algorithm/reduce.hpp
@brief hip reduce algorithms include file
*/

namespace tf::detail {

// ----------------------------------------------------------------------------
// reduction helper functions
// ----------------------------------------------------------------------------

/** @private */
template<unsigned nt, typename T>
struct hipBlockReduce {

  static const unsigned group_size = std::min(nt, hip_WARP_SIZE);
  static const unsigned num_passes = log2(group_size);
  static const unsigned num_items = nt / group_size;

  static_assert(
    nt && (0 == nt % hip_WARP_SIZE),
    "hipBlockReduce requires num threads to be a multiple of warp_size (32)"
  );

  /** @private */
  struct Storage {
    T data[std::max(nt, 2 * group_size)];
  };

  template<typename op_t>
  __device__ T operator()(unsigned, T, Storage&, unsigned, op_t, bool = true) const;
};

// function: reduce to be called from a block
template<unsigned nt, typename T>
template<typename op_t>
__device__ T hipBlockReduce<nt, T>::operator ()(
  unsigned tid, T x, Storage& storage, unsigned count, op_t op, bool ret
) const {

  // Store your data into shared memory.
  storage.data[tid] = x;
  __syncthreads();

  if(tid < group_size) {
    // Each thread scans within its lane.
    hip_strided_iterate<group_size, num_items>([&](auto i, auto j) {
      if(i > 0) {
        x = op(x, storage.data[j]);
      }
    }, tid, count);
    storage.data[tid] = x;
  }
  __syncthreads();

  auto count2 = count < group_size ? count : group_size;
  auto first = (1 & num_passes) ? group_size : 0;
  if(tid < group_size) {
    storage.data[first + tid] = x;
  }
  __syncthreads();

  hip_iterate<num_passes>([&](auto pass) {
    if(tid < group_size) {
      if(auto offset = 1 << pass; tid + offset < count2) {
        x = op(x, storage.data[first + offset + tid]);
      }
      first = group_size - first;
      storage.data[first + tid] = x;
    }
    __syncthreads();
  });

  if(ret) {
    x = storage.data[0];
    __syncthreads();
  }
  return x;
}

/** @private */
template <typename P, typename I, typename T, typename O>
void hip_reduce_loop(
  P&& p, I input, unsigned count, T* res, O op, void* ptr
) {

  using U = typename std::iterator_traits<I>::value_type;
  using E = std::decay_t<P>;

  auto buf = static_cast<U*>(ptr);
  auto B = (count + E::nv - 1) / E::nv;

  hip_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
    __shared__ typename hipBlockReduce<E::nt, U>::Storage shm;
    auto tile = hip_get_tile(bid, E::nv, count);
    auto x = hip_mem_to_reg_strided<E::nt, E::vt>(
      input + tile.begin, tid, tile.count()
    );
    // reduce multiple values per thread into a scalar.
    U s;
    hip_strided_iterate<E::nt, E::vt>(
      [&] (auto i, auto) { s = i ? op(s, x[i]) : x[0]; }, tid, tile.count()
    );
    // reduce to a scalar per block.
    s = hipBlockReduce<E::nt, U>()(
      tid, s, shm, (tile.count() < E::nt ? tile.count() : E::nt), op, false
    );
    if(!tid) {
      (1 == B) ? *res = op(*res, s) : buf[bid] = s;
    }
  });

  if(B > 1) {
    hip_reduce_loop(p, buf, B, res, op, buf+B);
  }
}

/** @private */
template <typename P, typename I, typename T, typename O>
void hip_uninitialized_reduce_loop(
  P&& p, I input, unsigned count, T* res, O op, void* ptr
) {

  using U = typename std::iterator_traits<I>::value_type;
  using E = std::decay_t<P>;

  auto buf = static_cast<U*>(ptr);
  auto B = (count + E::nv - 1) / E::nv;

  hip_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
    __shared__ typename hipBlockReduce<E::nt, U>::Storage shm;
    auto tile = hip_get_tile(bid, E::nv, count);
    auto x = hip_mem_to_reg_strided<E::nt, E::vt>(
      input + tile.begin, tid, tile.count()
    );
    // reduce multiple values per thread into a scalar.
    U s;
    hip_strided_iterate<E::nt, E::vt>(
      [&] (auto i, auto) { s = i ? op(s, x[i]) : x[0]; }, tid, tile.count()
    );
    // reduce to a scalar per block.
    s = hipBlockReduce<E::nt, U>()(
      tid, s, shm, (tile.count() < E::nt ? tile.count() : E::nt), op, false
    );
    if(!tid) {
      (1 == B) ? *res = s : buf[bid] = s;
    }
  });

  if(B > 1) {
    hip_uninitialized_reduce_loop(p, buf, B, res, op, buf+B);
  }
}

}  // namespace tf::detail ----------------------------------------------------

namespace tf {

/**
@brief queries the buffer size in bytes needed to call reduce kernels

@tparam P execution policy type
@tparam T value type

@param count number of elements to reduce

The function is used to allocate a buffer for calling tf::hip_reduce,
tf::hip_uninitialized_reduce, tf::hip_transform_reduce, and
tf::hip_transform_uninitialized_reduce.
*/
template <typename P, typename T>
unsigned hip_reduce_buffer_size(unsigned count) {
  using E = std::decay_t<P>;
  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned n = 0;
  for(auto b=B; b>1; n += (b=(b+E::nv-1)/E::nv));
  return n*sizeof(T);
}

// ----------------------------------------------------------------------------
// hip_reduce
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous parallel reduction over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements
@param buf pointer to the temporary buffer

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first != last) {
  *result = op(*result, *first++);
}
@endcode
 */
template <typename P, typename I, typename T, typename O>
void hip_reduce(
  P&& p, I first, I last, T* res, O op, void* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::hip_reduce_loop(p, first, count, res, op, buf);
}

// ----------------------------------------------------------------------------
// hip_uninitialized_reduce
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous parallel reduction over a range of items without
       an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param op binary operator to apply to reduce elements
@param buf pointer to the temporary buffer

This method is equivalent to the parallel execution of the following loop
on a GPU:

@code{.cpp}
*result = *first++;  // no initial values partitipcate in the loop
while (first != last) {
  *result = op(*result, *first++);
}
@endcode
*/
template <typename P, typename I, typename T, typename O>
void hip_uninitialized_reduce(
  P&& p, I first, I last, T* res, O op, void* buf
) {
  unsigned count = std::distance(first, last);
  if(count == 0) {
    return;
  }
  detail::hip_uninitialized_reduce_loop(p, first, count, res, op, buf);
}

// ----------------------------------------------------------------------------
// transform_reduce
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous parallel reduction over a range of transformed items
       without an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param bop binary operator to apply to reduce elements
@param uop unary operator to apply to transform elements
@param buf pointer to the temporary buffer

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first != last) {
  *result = bop(*result, uop(*first++));
}
@endcode
*/
template<typename P, typename I, typename T, typename O, typename U>
void hip_transform_reduce(
  P&& p, I first, I last, T* res, O bop, U uop, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // reduction loop
  detail::hip_reduce_loop(p,
    hip_make_load_iterator<T>([=]__device__(auto i){
      return uop(*(first+i));
    }),
    count, res, bop, buf
  );
}

// ----------------------------------------------------------------------------
// transform_uninitialized_reduce
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous parallel reduction over a range of transformed items
       with an initial value

@tparam P execution policy type
@tparam I input iterator type
@tparam T value type
@tparam O binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param res pointer to the result
@param bop binary operator to apply to reduce elements
@param uop unary operator to apply to transform elements
@param buf pointer to the temporary buffer

This method is equivalent to the parallel execution of the following loop
on a GPU:

@code{.cpp}
*result = uop(*first++);  // no initial values partitipcate in the loop
while (first != last) {
  *result = bop(*result, uop(*first++));
}
@endcode
*/
template<typename P, typename I, typename T, typename O, typename U>
void hip_transform_uninitialized_reduce(
  P&& p, I first, I last, T* res, O bop, U uop, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // reduction loop
  //detail::hip_transform_reduce_loop(
  //  p, first, count, res, bop, uop, false, s, buf
  //);
  detail::hip_uninitialized_reduce_loop(p,
    hip_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
    count, res, bop, buf
  );
}

// ----------------------------------------------------------------------------

//template <typename T, typename C>
//__device__ void hip_warp_reduce(
//  volatile T* shm, size_t N, size_t tid, C op
//) {
//  if(tid + 32 < N) shm[tid] = op(shm[tid], shm[tid+32]);
//  if(tid + 16 < N) shm[tid] = op(shm[tid], shm[tid+16]);
//  if(tid +  8 < N) shm[tid] = op(shm[tid], shm[tid+8]);
//  if(tid +  4 < N) shm[tid] = op(shm[tid], shm[tid+4]);
//  if(tid +  2 < N) shm[tid] = op(shm[tid], shm[tid+2]);
//  if(tid +  1 < N) shm[tid] = op(shm[tid], shm[tid+1]);
//}
//
//template <typename I, typename T, typename C, bool uninitialized>
//__global__ void hip_reduce(I first, size_t N, T* res, C op) {
//
//  size_t tid = threadIdx.x;
//
//  if(tid >= N) {
//    return;
//  }
//
//  hipSharedMemory<T> shared_memory;
//  T* shm = shared_memory.get();
//
//  shm[tid] = *(first+tid);
//
//  for(size_t i=tid+blockDim.x; i<N; i+=blockDim.x) {
//    shm[tid] = op(shm[tid], *(first+i));
//  }
//
//  __syncthreads();
//
//  for(size_t s = blockDim.x / 2; s > 32; s >>= 1) {
//    if(tid < s && tid + s < N) {
//      shm[tid] = op(shm[tid], shm[tid+s]);
//    }
//    __syncthreads();
//  }
//
//  if(tid < 32) {
//    hip_warp_reduce(shm, N, tid, op);
//  }
//
//  if(tid == 0) {
//    if constexpr (uninitialized) {
//      *res = shm[0];
//    }
//    else {
//      *res = op(*res, shm[0]);
//    }
//  }
//}

// ----------------------------------------------------------------------------
// hipFlowCapturer
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename C>
hipTask hipFlowCapturer::reduce(I first, I last, T* result, C c) {

  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_reduce(p, first, last, result, c, buf.get().data());
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
hipTask hipFlowCapturer::uninitialized_reduce(I first, I last, T* result, C c) {

  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_uninitialized_reduce(p, first, last, result, c, buf.get().data());
  });
}

// Function: transform_reduce
template <typename I, typename T, typename C, typename U>
hipTask hipFlowCapturer::transform_reduce(
  I first, I last, T* result, C bop, U uop
) {

  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_reduce(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename C, typename U>
hipTask hipFlowCapturer::transform_uninitialized_reduce(
  I first, I last, T* result, C bop, U uop) {

  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_uninitialized_reduce(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}

// Function: reduce
template <typename I, typename T, typename C>
void hipFlowCapturer::reduce(
  hipTask task, I first, I last, T* result, C c
) {

  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_reduce(p, first, last, result, c, buf.get().data());
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
void hipFlowCapturer::uninitialized_reduce(
  hipTask task, I first, I last, T* result, C c
) {
  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_uninitialized_reduce(p, first, last, result, c, buf.get().data());
  });
}

// Function: transform_reduce
template <typename I, typename T, typename C, typename U>
void hipFlowCapturer::transform_reduce(
  hipTask task, I first, I last, T* result, C bop, U uop
) {

  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_reduce(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename C, typename U>
void hipFlowCapturer::transform_uninitialized_reduce(
  hipTask task, I first, I last, T* result, C bop, U uop
) {

  // TODO
  auto bufsz = hip_reduce_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_uninitialized_reduce(
      p, first, last, result, bop, uop, buf.get().data()
    );
  });
}


// ----------------------------------------------------------------------------
// hipFlow
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename B>
hipTask hipFlow::reduce(I first, I last, T* result, B bop) {
  return capture([=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.reduce(first, last, result, bop);
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename B>
hipTask hipFlow::uninitialized_reduce(I first, I last, T* result, B bop) {
  return capture([=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.uninitialized_reduce(first, last, result, bop);
  });
}

// Function: transform_reduce
template <typename I, typename T, typename B, typename U>
hipTask hipFlow::transform_reduce(I first, I last, T* result, B bop, U uop) {
  return capture([=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_reduce(first, last, result, bop, uop);
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename B, typename U>
hipTask hipFlow::transform_uninitialized_reduce(
  I first, I last, T* result, B bop, U uop
) {
  return capture([=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_uninitialized_reduce(first, last, result, bop, uop);
  });
}

// Function: reduce
template <typename I, typename T, typename C>
void hipFlow::reduce(hipTask task, I first, I last, T* result, C op) {
  capture(task, [=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.reduce(first, last, result, op);
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
void hipFlow::uninitialized_reduce(
  hipTask task, I first, I last, T* result, C op
) {
  capture(task, [=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.uninitialized_reduce(first, last, result, op);
  });
}

// Function: transform_reduce
template <typename I, typename T, typename B, typename U>
void hipFlow::transform_reduce(
  hipTask task, I first, I last, T* result, B bop, U uop
) {
  capture(task, [=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_reduce(first, last, result, bop, uop);
  });
}

// Function: transform_uninitialized_reduce
template <typename I, typename T, typename B, typename U>
void hipFlow::transform_uninitialized_reduce(
  hipTask task, I first, I last, T* result, B bop, U uop
) {
  capture(task, [=](hipFlowCapturer& cap){
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_uninitialized_reduce(first, last, result, bop, uop);
  });
}


}  // end of namespace tf -----------------------------------------------------

