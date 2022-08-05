#pragma once

#include "hip_error.hpp"

/**
@file hip_execution_policy.hpp
@brief hip execution policy include file
*/

namespace tf {

/**
@class hipExecutionPolicy

@brief class to define execution policy for hip standard algorithms

@tparam NT number of threads per block
@tparam VT number of work units per thread

Execution policy configures the kernel execution parameters in hip algorithms.
The first template argument, @c NT, the number of threads per block should
always be a power-of-two number.
The second template argument, @c VT, the number of work units per thread
is recommended to be an odd number to avoid bank conflict.

Details can be referred to @ref hipSTDExecutionPolicy.
*/
template<unsigned NT, unsigned VT>
class hipExecutionPolicy {

  static_assert(is_pow2(NT), "max # threads per block must be a power of two");

  public:

  /** @brief static constant for getting the number of threads per block */
  const static unsigned nt = NT;

  /** @brief static constant for getting the number of work units per thread */
  const static unsigned vt = VT;

  /** @brief static constant for getting the number of elements to process per block */
  const static unsigned nv = NT*VT;

  /**
  @brief constructs an execution policy object with default stream
   */
  hipExecutionPolicy() = default;

  /**
  @brief constructs an execution policy object with the given stream
   */
  explicit hipExecutionPolicy(hipStream_t s) : _stream{s} {}
  
  /**
  @brief queries the associated stream
   */
  hipStream_t stream() noexcept { return _stream; };

  /**
  @brief assigns a stream
   */
  void stream(hipStream_t stream) noexcept { _stream = stream; }

  private:

  hipStream_t _stream {0};
};

/**
@brief default execution policy
 */
using hipDefaultExecutionPolicy = hipExecutionPolicy<512, 9>;

}  // end of namespace tf -----------------------------------------------------



