#pragma once

#include "../hipflow.hpp"

/**
@file taskflow/hip/algorithm/transform.hpp
@brief hip parallel-transform algorithms include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// transform
// ----------------------------------------------------------------------------

namespace detail {

/** @private */
template <typename P, typename I, typename O, typename C>
void hip_transform_loop(P&& p, I first, unsigned count, O output, C op) {

  using E = std::decay_t<P>;

  unsigned B = (count + E::nv - 1) / E::nv;

  hip_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
    auto tile = hip_get_tile(bid, E::nv, count);
    hip_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
      auto offset = j + tile.begin;
      *(output + offset) = op(*(first+offset));
    }, tid, tile.count());
  });
}

/** @private */
template <typename P, typename I1, typename I2, typename O, typename C>
void hip_transform_loop(
  P&& p, I1 first1, I2 first2, unsigned count, O output, C op
) {

  using E = std::decay_t<P>;

  unsigned B = (count + E::nv - 1) / E::nv;

  hip_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
    auto tile = hip_get_tile(bid, E::nv, count);
    hip_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
      auto offset = j + tile.begin;
      *(output + offset) = op(*(first1+offset), *(first2+offset));
    }, tid, tile.count());
  });
}

}  // end of namespace detail -------------------------------------------------

// ----------------------------------------------------------------------------
// hip standard algorithms: transform
// ----------------------------------------------------------------------------

/**
@brief performs asynchronous parallel transforms over a range of items

@tparam P execution policy type
@tparam I input iterator type
@tparam O output iterator type
@tparam C unary operator type

@param p execution policy
@param first iterator to the beginning of the range
@param last iterator to the end of the range
@param output iterator to the beginning of the output range
@param op unary operator to apply to transform each item

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first != last) {
  *output++ = op(*first++);
}
@endcode

*/
template <typename P, typename I, typename O, typename C>
void hip_transform(P&& p, I first, I last, O output, C op) {

  unsigned count = std::distance(first, last);

  if(count == 0) {
    return;
  }

  detail::hip_transform_loop(p, first, count, output, op);
}

/**
@brief performs asynchronous parallel transforms over two ranges of items

@tparam P execution policy type
@tparam I1 first input iterator type
@tparam I2 second input iterator type
@tparam O output iterator type
@tparam C binary operator type

@param p execution policy
@param first1 iterator to the beginning of the first range
@param last1 iterator to the end of the first range
@param first2 iterator to the beginning of the second range
@param output iterator to the beginning of the output range
@param op binary operator to apply to transform each pair of items

This method is equivalent to the parallel execution of the following loop on a GPU:

@code{.cpp}
while (first1 != last1) {
  *output++ = op(*first1++, *first2++);
}
@endcode
*/
template <typename P, typename I1, typename I2, typename O, typename C>
void hip_transform(
  P&& p, I1 first1, I1 last1, I2 first2, O output, C op
) {

  unsigned count = std::distance(first1, last1);

  if(count == 0) {
    return;
  }

  detail::hip_transform_loop(p, first1, first2, count, output, op);
}

// ----------------------------------------------------------------------------
// hipFlow
// ----------------------------------------------------------------------------

// Function: transform
template <typename I, typename O, typename C>
hipTask hipFlow::transform(I first, I last, O output, C c) {
  return capture([=](hipFlowCapturer& cap) mutable {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform(first, last, output, c);
  });
}

// Function: transform
template <typename I1, typename I2, typename O, typename C>
hipTask hipFlow::transform(I1 first1, I1 last1, I2 first2, O output, C c) {
  return capture([=](hipFlowCapturer& cap) mutable {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform(first1, last1, first2, output, c);
  });
}

// Function: update transform
template <typename I, typename O, typename C>
void hipFlow::transform(hipTask task, I first, I last, O output, C c) {
  capture(task, [=](hipFlowCapturer& cap) mutable {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform(first, last, output, c);
  });
}

// Function: update transform
template <typename I1, typename I2, typename O, typename C>
void hipFlow::transform(
  hipTask task, I1 first1, I1 last1, I2 first2, O output, C c
) {
  capture(task, [=](hipFlowCapturer& cap) mutable {
    cap.make_optimizer<hipLinearCapturing>();
    cap.transform(first1, last1, first2, output, c);
  });
}

// ----------------------------------------------------------------------------
// hipFlowCapturer
// ----------------------------------------------------------------------------

// Function: transform
template <typename I, typename O, typename C>
hipTask hipFlowCapturer::transform(I first, I last, O output, C op) {
  return on([=](hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform(p, first, last, output, op);
  });
}

// Function: transform
template <typename I1, typename I2, typename O, typename C>
hipTask hipFlowCapturer::transform(
  I1 first1, I1 last1, I2 first2, O output, C op
) {
  return on([=](hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform(p, first1, last1, first2, output, op);
  });
}

// Function: transform
template <typename I, typename O, typename C>
void hipFlowCapturer::transform(
  hipTask task, I first, I last, O output, C op
) {
  on(task, [=] (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform(p, first, last, output, op);
  });
}

// Function: transform
template <typename I1, typename I2, typename O, typename C>
void hipFlowCapturer::transform(
  hipTask task, I1 first1, I1 last1, I2 first2, O output, C op
) {
  on(task, [=] (hipStream_t stream) mutable {
    hipDefaultExecutionPolicy p(stream);
    hip_transform(p, first1, last1, first2, output, op);
  });
}

}  // end of namespace tf -----------------------------------------------------






