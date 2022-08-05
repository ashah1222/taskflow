#pragma once

#include "reduce.hpp"

/**
@file taskflow/hip/algorithm/scan.hpp
@brief hip scan algorithm include file
*/

namespace tf::detail {

// ----------------------------------------------------------------------------
// scan
// ----------------------------------------------------------------------------

/** @private */
inline constexpr unsigned hipScanRecursionThreshold = 8;

/** @private */
enum class hipScanType : int {
  EXCLUSIVE = 1,
  INCLUSIVE
};

/** @private */
template<typename T, unsigned vt = 0, bool is_array = (vt > 0)>
struct hipScanResult {
  T scan;
  T reduction;
};

/** @private */
template<typename T, unsigned vt>
struct hipScanResult<T, vt, true> {
  hipArray<T, vt> scan;
  T reduction;
};

//-----------------------------------------------------------------------------

/** @private */
template<unsigned nt, typename T>
struct hipBlockScan {

  const static unsigned num_warps  = nt / hip_WARP_SIZE;
  const static unsigned num_passes = log2(nt);
  const static unsigned capacity   = nt + num_warps;

  /** @private */
  union storage_t {
    T data[2 * nt];
    struct { T threads[nt], warps[num_warps]; };
  };

  // standard scan
  template<typename op_t>
  __device__ hipScanResult<T> operator ()(
    unsigned tid,
    T x,
    storage_t& storage,
    unsigned count = nt,
    op_t op = op_t(),
    T init = T(),
    hipScanType type = hipScanType::EXCLUSIVE
  ) const;

  // vectorized scan. accepts multiple values per thread and adds in
  // optional global carry-in
  template<unsigned vt, typename op_t>
  __device__ hipScanResult<T, vt> operator()(
    unsigned tid,
    hipArray<T, vt> x,
    storage_t& storage,
    T carry_in = T(),
    bool use_carry_in = false,
    unsigned count = nt,
    op_t op = op_t(),
    T init = T(),
    hipScanType type = hipScanType::EXCLUSIVE
  ) const;
};

// standard scan
template <unsigned nt, typename T>
template<typename op_t>
__device__ hipScanResult<T> hipBlockScan<nt, T>::operator () (
  unsigned tid, T x, storage_t& storage, unsigned count, op_t op,
  T init, hipScanType type
) const {

  unsigned first = 0;
  storage.data[first + tid] = x;
  __syncthreads();

  hip_iterate<num_passes>([&](auto pass) {
    if(auto offset = 1<<pass; tid >= offset) {
      x = op(storage.data[first + tid - offset], x);
    }
    first = nt - first;
    storage.data[first + tid] = x;
    __syncthreads();
  });

  hipScanResult<T> result;
  result.reduction = storage.data[first + count - 1];
  result.scan = (tid < count) ?
    (hipScanType::INCLUSIVE == type ? x :
      (tid ? storage.data[first + tid - 1] : init)) :
    result.reduction;
  __syncthreads();

  return result;
}

// vectorized scan block
template <unsigned nt, typename T>
template<unsigned vt, typename op_t>
__device__ hipScanResult<T, vt> hipBlockScan<nt, T>::operator()(
  unsigned tid,
  hipArray<T, vt> x,
  storage_t& storage,
  T carry_in,
  bool use_carry_in,
  unsigned count, op_t op,
  T init,
  hipScanType type
) const {

  // Start with an inclusive scan of the in-range elements.
  if(count >= nt * vt) {
    hip_iterate<vt>([&](auto i) {
      x[i] = i ? op(x[i], x[i - 1]) : x[i];
    });
  } else {
    hip_iterate<vt>([&](auto i) {
      auto index = vt * tid + i;
      x[i] = i ?
        ((index < count) ? op(x[i], x[i - 1]) : x[i - 1]) :
        (x[i] = (index < count) ? x[i] : init);
    });
  }

  // Scan the thread-local reductions for a carry-in for each thread.
  auto result = operator()(
    tid, x[vt - 1], storage,
    (count + vt - 1) / vt, op, init, hipScanType::EXCLUSIVE
  );

  // Perform the scan downsweep and add both the global carry-in and the
  // thread carry-in to the values.
  if(use_carry_in) {
    result.reduction = op(carry_in, result.reduction);
    result.scan = tid ? op(carry_in, result.scan) : carry_in;
  } else {
    use_carry_in = tid > 0;
  }

  hipArray<T, vt> y;
  hip_iterate<vt>([&](auto i) {
    if(hipScanType::EXCLUSIVE == type) {
      y[i] = i ? x[i - 1] : result.scan;
      if(use_carry_in && i > 0) y[i] = op(result.scan, y[i]);
    } else {
      y[i] = use_carry_in ? op(x[i], result.scan) : x[i];
    }
  });

  return hipScanResult<T, vt> { y, result.reduction };
}

/**
@private
@brief single-pass scan for small input
 */
template <typename P, typename I, typename O, typename C>
void hip_single_pass_scan(
  P&& p,
  hipScanType scan_type,
  I input,
  unsigned count,
  O output,
  C op
  //reduction_it reduction,
) {

  using T = typename std::iterator_traits<O>::value_type;
  using E = std::decay_t<P>;

  // Small input specialization. This is the non-recursive branch.
  hip_kernel<<<1, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {

    using scan_t = hipBlockScan<E::nt, T>;

    __shared__ union {
      typename scan_t::storage_t scan;
      T values[E::nv];
    } shared;

    auto carry_in = T();
    for(unsigned cur = 0; cur < count; cur += E::nv) {
      // Cooperatively load values into register.
      auto count2 = min(count - cur, E::nv);

      auto x = hip_mem_to_reg_thread<E::nt, E::vt>(input + cur,
        tid, count2, shared.values);

      auto result = scan_t()(tid, x, shared.scan,
        carry_in, cur > 0, count2, op, T(), scan_type);

      // Store the scanned values back to global memory.
      hip_reg_to_mem_thread<E::nt, E::vt>(result.scan, tid, count2,
        output + cur, shared.values);

      // Roll the reduction into carry_in.
      carry_in = result.reduction;
    }

    // Store the carry-out to the reduction pointer. This may be a
    // discard_iterator_t if no reduction is wanted.
    //if(!tid) *reduction = carry_in;
  });
}

/**
@private

@brief main scan loop
*/
template<typename P, typename I, typename O, typename C>
void hip_scan_loop(
  P&& p,
  hipScanType scan_type,
  I input,
  unsigned count,
  O output,
  C op,
  //reduction_it reduction,
  void* ptr
) {

  using E = std::decay_t<P>;
  using T = typename std::iterator_traits<O>::value_type;

  T* buffer = static_cast<T*>(ptr);

  //launch_t::cta_dim(context).B(count);
  unsigned B = (count + E::nv - 1) / E::nv;

  if(B > hipScanRecursionThreshold) {

    //hipDeviceVector<T> partials(B);
    //auto buffer = partials.data();

    // upsweep phase
    hip_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {

      __shared__ typename hipBlockReduce<E::nt, T>::Storage shm;

      // Load the tile's data into register.
      auto tile = hip_get_tile(bid, E::nv, count);
      auto x = hip_mem_to_reg_strided<E::nt, E::vt>(
        input + tile.begin, tid, tile.count()
      );

      // Reduce the thread's values into a scalar.
      T scalar;
      hip_strided_iterate<E::nt, E::vt>(
        [&] (auto i, auto j) { scalar = i ? op(scalar, x[i]) : x[0]; },
        tid, tile.count()
      );

      // Reduce across all threads.
      auto all_reduce = hipBlockReduce<E::nt, T>()(
        tid, scalar, shm, tile.count(), op
      );

      // Store the final reduction to the partials.
      if(!tid) {
        buffer[bid] = all_reduce;
      }
    });

    // recursively call scan
    //hip_scan_loop(p, hipScanType::EXCLUSIVE, buffer, B, buffer, op, S);
    hip_scan_loop(
      p, hipScanType::EXCLUSIVE, buffer, B, buffer, op, buffer+B
    );

    // downsweep: perform an intra-tile scan and add the scan of the partials
    // as carry-in
    hip_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {

      using scan_t = hipBlockScan<E::nt, T>;

      __shared__ union {
        typename scan_t::storage_t scan;
        T values[E::nv];
      } shared;

      // Load a tile to register in thread order.
      auto tile = hip_get_tile(bid, E::nv, count);
      auto x = hip_mem_to_reg_thread<E::nt, E::vt>(
        input + tile.begin, tid, tile.count(), shared.values
      );

      // Scan the array with carry-in from the partials.
      auto y = scan_t()(tid, x, shared.scan,
        buffer[bid], bid > 0, tile.count(), op, T(),
        scan_type).scan;

      // Store the scanned values to the output.
      hip_reg_to_mem_thread<E::nt, E::vt>(
        y, tid, tile.count(), output + tile.begin, shared.values
      );
    });
  }
  // Small input specialization. This is the non-recursive branch.
  else {
    hip_single_pass_scan(p, scan_type, input, count, output, op);
  }
}

}  // namespace tf::detail ----------------------------------------------------

namespace tf {

/**
@brief queries the buffer size in bytes needed to call scan kernels

@tparam P execution policy type
@tparam T value type

@param count number of elements to scan

The function is used to allocate a buffer for calling
tf::hip_inclusive_scan, tf::hip_exclusive_scan,
tf::hip_transform_inclusive_scan, and tf::hip_transform_exclusive_scan.
*/
template <typename P, typename T>
unsigned hip_scan_buffer_size(unsigned count) {
  using E = std::decay_t<P>;
  unsigned B = (count + E::nv - 1) / E::nv;
  unsigned n = 0;
  for(auto b=B; b>detail::hipScanRecursionThreshold; b=(b+E::nv-1)/E::nv) {
    n += b;
  }
  return n*sizeof(T);
}

// ----------------------------------------------------------------------------
// inclusive scan
// ----------------------------------------------------------------------------

//template<typename P, typename I, typename O, typename C>
//void hip_inclusive_scan(P&& p, I first, I last, O output, C op) {
//
//  unsigned count = std::distance(first, last);
//
//  if(count == 0) {
//    return;
//  }
//
//  using T = typename std::iterator_traits<O>::value_type;
//
//  // allocate temporary buffer
//  hipDeviceVector<std::byte> temp(hip_scan_buffer_size<P, T>(count));
//  
//  // launch the scan loop
//  detail::hip_scan_loop(
//    p, detail::hipScanType::INCLUSIVE, first, count, output, op, temp.data()
//  );
//
//  // synchronize the execution
//  p.synchronize();
//}

/**
@brief performs asynchronous inclusive scan over a range of items

@tparam P execution policy type
@tparam I input iterator
@tparam O output iterator
@tparam C binary operator type

@param p execution policy
@param first iterator to the beginning of the input range
@param last iterator to the end of the input range
@param output iterator to the beginning of the output range
@param op binary operator to apply to scan
@param buf pointer to the temporary buffer

*/
template<typename P, typename I, typename O, typename C>
void hip_inclusive_scan(
  P&& p, I first, I last, O output, C op, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // launch the scan loop
  detail::hip_scan_loop(
    p, detail::hipScanType::INCLUSIVE, first, count, output, op, buf
  );
}

// ----------------------------------------------------------------------------
// transform inclusive_scan
// ----------------------------------------------------------------------------

//template<typename P, typename I, typename O, typename C, typename U>
//void hip_transform_inclusive_scan(
//  P&& p, I first, I last, O output, C bop, U uop
//) {
//
//  unsigned count = std::distance(first, last);
//
//  if(count == 0) {
//    return;
//  }
//
//  using T = typename std::iterator_traits<O>::value_type;
//
//  // allocate temporary buffer
//  hipDeviceVector<std::byte> temp(hip_scan_buffer_size<P, T>(count));
//  auto buf = temp.data();
//
//  // launch the scan loop
//  detail::hip_scan_loop(
//    p, detail::hipScanType::INCLUSIVE,
//    hip_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
//    count, output, bop, buf
//  );
//
//  // synchronize the execution
//  p.synchronize();
//}

/**
@brief performs asynchronous inclusive scan over a range of transformed items

@tparam P execution policy type
@tparam I input iterator
@tparam O output iterator
@tparam C binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the input range
@param last iterator to the end of the input range
@param output iterator to the beginning of the output range
@param bop binary operator to apply to scan
@param uop unary operator to apply to transform each item before scan
@param buf pointer to the temporary buffer

*/
template<typename P, typename I, typename O, typename C, typename U>
void hip_transform_inclusive_scan(
  P&& p, I first, I last, O output, C bop, U uop, void* buf
) {

  using T = typename std::iterator_traits<O>::value_type;

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // launch the scan loop
  detail::hip_scan_loop(
    p, detail::hipScanType::INCLUSIVE,
    hip_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
    count, output, bop, buf
  );
}

// ----------------------------------------------------------------------------
// exclusive scan
// ----------------------------------------------------------------------------

//template<typename P, typename I, typename O, typename C>
//void hip_exclusive_scan(P&& p, I first, I last, O output, C op) {
//
//  unsigned count = std::distance(first, last);
//
//  if(count == 0) {
//    return;
//  }
//
//  using T = typename std::iterator_traits<O>::value_type;
//
//  // allocate temporary buffer
//  hipDeviceVector<std::byte> temp(hip_scan_buffer_size<P, T>(count));
//  auto buf = temp.data();
//
//  // launch the scan loop
//  detail::hip_scan_loop(
//    p, detail::hipScanType::EXCLUSIVE, first, count, output, op, buf
//  );
//
//  // synchronize the execution
//  p.synchronize();
//}

/**
@brief performs asynchronous exclusive scan over a range of items

@tparam P execution policy type
@tparam I input iterator
@tparam O output iterator
@tparam C binary operator type

@param p execution policy
@param first iterator to the beginning of the input range
@param last iterator to the end of the input range
@param output iterator to the beginning of the output range
@param op binary operator to apply to scan
@param buf pointer to the temporary buffer

*/
template<typename P, typename I, typename O, typename C>
void hip_exclusive_scan(
  P&& p, I first, I last, O output, C op, void* buf
) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // launch the scan loop
  detail::hip_scan_loop(
    p, detail::hipScanType::EXCLUSIVE, first, count, output, op, buf
  );
}

// ----------------------------------------------------------------------------
// transform exclusive scan
// ----------------------------------------------------------------------------

//template<typename P, typename I, typename O, typename C, typename U>
//void hip_transform_exclusive_scan(
//  P&& p, I first, I last, O output, C bop, U uop
//) {
//
//  unsigned count = std::distance(first, last);
//
//  if(count == 0) {
//    return;
//  }
//
//  using T = typename std::iterator_traits<O>::value_type;
//
//  // allocate temporary buffer
//  hipDeviceVector<std::byte> temp(hip_scan_buffer_size<P, T>(count));
//  auto buf = temp.data();
//
//  // launch the scan loop
//  detail::hip_scan_loop(
//    p, detail::hipScanType::EXCLUSIVE,
//    hip_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
//    count, output, bop, buf
//  );
//
//  // synchronize the execution
//  p.synchronize();
//}

/**
@brief performs asynchronous exclusive scan over a range of items

@tparam P execution policy type
@tparam I input iterator
@tparam O output iterator
@tparam C binary operator type
@tparam U unary operator type

@param p execution policy
@param first iterator to the beginning of the input range
@param last iterator to the end of the input range
@param output iterator to the beginning of the output range
@param bop binary operator to apply to scan
@param uop unary operator to apply to transform each item before scan
@param buf pointer to the temporary buffer

*/
template<typename P, typename I, typename O, typename C, typename U>
void hip_transform_exclusive_scan(
  P&& p, I first, I last, O output, C bop, U uop, void* buf
) {

  using T = typename std::iterator_traits<O>::value_type;

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  // launch the scan loop
  detail::hip_scan_loop(
    p, detail::hipScanType::EXCLUSIVE,
    hip_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
    count, output, bop, buf
  );
}

// ----------------------------------------------------------------------------
// hipFlow
// ----------------------------------------------------------------------------

// Function: inclusive_scan
template <typename I, typename O, typename C>
hipTask hipFlow::inclusive_scan(I first, I last, O output, C op) {
  return capture([=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.inclusive_scan(first, last, output, op);
  });
}

// Function: inclusive_scan
template <typename I, typename O, typename C>
void hipFlow::inclusive_scan(hipTask task, I first, I last, O output, C op) {
  capture(task, [=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.inclusive_scan(first, last, output, op);
  });
}

// Function: exclusive_scan
template <typename I, typename O, typename C>
hipTask hipFlow::exclusive_scan(I first, I last, O output, C op) {
  return capture([=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.exclusive_scan(first, last, output, op);
  });
}

// Function: exclusive_scan
template <typename I, typename O, typename C>
void hipFlow::exclusive_scan(hipTask task, I first, I last, O output, C op) {
  capture(task, [=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.exclusive_scan(first, last, output, op);
  });
}

// Function: transform_inclusive_scan
template <typename I, typename O, typename B, typename U>
hipTask hipFlow::transform_inclusive_scan(
  I first, I last, O output, B bop, U uop
) {
  return capture([=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_inclusive_scan(first, last, output, bop, uop);
  });
}

// Function: transform_inclusive_scan
template <typename I, typename O, typename B, typename U>
void hipFlow::transform_inclusive_scan(
  hipTask task, I first, I last, O output, B bop, U uop
) {
  capture(task, [=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_inclusive_scan(first, last, output, bop, uop);
  });
}

// Function: transform_exclusive_scan
template <typename I, typename O, typename B, typename U>
hipTask hipFlow::transform_exclusive_scan(
  I first, I last, O output, B bop, U uop
) {
  return capture([=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_exclusive_scan(first, last, output, bop, uop);
  });
}

// Function: transform_exclusive_scan
template <typename I, typename O, typename B, typename U>
void hipFlow::transform_exclusive_scan(
  hipTask task, I first, I last, O output, B bop, U uop
) {
  capture(task, [=](hipFlowCapturer& cap) {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform_exclusive_scan(first, last, output, bop, uop);
  });
}

// ----------------------------------------------------------------------------
// hipFlowCapturer
// ----------------------------------------------------------------------------

// Function: inclusive_scan
template <typename I, typename O, typename C>
hipTask hipFlowCapturer::inclusive_scan(I first, I last, O output, C op) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_inclusive_scan(p, first, last, output, op, buf.get().data());
  });
}

// Function: inclusive_scan
template <typename I, typename O, typename C>
void hipFlowCapturer::inclusive_scan(
  hipTask task, I first, I last, O output, C op
) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_inclusive_scan(p, first, last, output, op, buf.get().data());
  });
}

// Function: exclusive_scan
template <typename I, typename O, typename C>
hipTask hipFlowCapturer::exclusive_scan(I first, I last, O output, C op) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_exclusive_scan(p, first, last, output, op, buf.get().data());
  });
}

// Function: exclusive_scan
template <typename I, typename O, typename C>
void hipFlowCapturer::exclusive_scan(
  hipTask task, I first, I last, O output, C op
) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_exclusive_scan(p, first, last, output, op, buf.get().data());
  });
}

// Function: transform_inclusive_scan
template <typename I, typename O, typename B, typename U>
hipTask hipFlowCapturer::transform_inclusive_scan(
  I first, I last, O output, B bop, U uop
) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_inclusive_scan(
      p, first, last, output, bop, uop, buf.get().data()
    );
  });
}

// Function: transform_inclusive_scan
template <typename I, typename O, typename B, typename U>
void hipFlowCapturer::transform_inclusive_scan(
  hipTask task, I first, I last, O output, B bop, U uop
) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_inclusive_scan(
      p, first, last, output, bop, uop, buf.get().data()
    );
  });
}

// Function: transform_exclusive_scan
template <typename I, typename O, typename B, typename U>
hipTask hipFlowCapturer::transform_exclusive_scan(
  I first, I last, O output, B bop, U uop
) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  return on([=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_exclusive_scan(
      p, first, last, output, bop, uop, buf.get().data()
    );
  });
}

// Function: transform_exclusive_scan
template <typename I, typename O, typename B, typename U>
void hipFlowCapturer::transform_exclusive_scan(
  hipTask task, I first, I last, O output, B bop, U uop
) {

  using T = typename std::iterator_traits<O>::value_type;

  auto bufsz = hip_scan_buffer_size<hipDefaultExecutionPolicy, T>(
    std::distance(first, last)
  );

  on(task, [=, buf=MoC{hipDeviceVector<std::byte>(bufsz)}] 
  (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform_exclusive_scan(
      p, first, last, output, bop, uop, buf.get().data()
    );
  });
}


}  // end of namespace tf -----------------------------------------------------



