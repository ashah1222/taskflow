#pragma once

#include <hip.h>
#include <iostream>
#include <sstream>
#include <exception>

#include "../utility/stream.hpp"

#define TF_HIP_EXPAND( x ) x
#define TF_HIP_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
#define TF_HIP_REMOVE_FIRST(...) TF_HIP_EXPAND(TF_HIP_REMOVE_FIRST_HELPER(__VA_ARGS__))
#define TF_HIP_GET_FIRST_HELPER(N, ...) N
#define TF_HIP_GET_FIRST(...) TF_HIP_EXPAND(TF_HIP_GET_FIRST_HELPER(__VA_ARGS__))

#define TF_CHECK_HIP(...)                                       \
if(TF_HIP_GET_FIRST(__VA_ARGS__) != hipSuccess) {              \
  std::ostringstream oss;                                        \
  auto __ev__ = TF_HIP_GET_FIRST(__VA_ARGS__);                  \
  oss << "[" << __FILE__ << ":" << __LINE__ << "] "              \
      << (hipGetErrorString(__ev__)) << " ("                    \
      << (hipGetErrorName(__ev__)) << ") - ";                   \
  tf::ostreamize(oss, TF_HIP_REMOVE_FIRST(__VA_ARGS__));        \
  throw std::runtime_error(oss.str());                           \
}

