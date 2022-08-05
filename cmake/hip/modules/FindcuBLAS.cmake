
# ==================================================================================================
# This file is part of the cuBLASt project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================
#
# Defines the following variables:
#   CUBLAS_FOUND          Boolean holding whether or not the cuBLAS library was found
#   CUBLAS_INCLUDE_DIRS   The HIP and cuBLAS include directory
#   HIP_LIBRARIES        The HIP library
#   CUBLAS_LIBRARIES      The cuBLAS library
#
# In case HIP is not installed in the default directory, set the HIP_ROOT variable to point to
# the root of cuBLAS, such that 'cublas_v2.h' can be found in $HIP_ROOT/include. This can either be
# done using an environmental variable (e.g. export HIP_ROOT=/path/to/cuBLAS) or using a CMake
# variable (e.g. cmake -DHIP_ROOT=/path/to/cuBLAS ..).
#
# ==================================================================================================

# Sets the possible install locations
set(CUBLAS_HINTS
  ${HIP_ROOT}
  $ENV{HIP_ROOT}
  $ENV{HIP_TOOLKIT_ROOT_DIR}
)
set(CUBLAS_PATHS
  /usr
  /usr/local
  /usr/local/hip
)

# Finds the include directories
find_path(CUBLAS_INCLUDE_DIRS
  NAMES cublas_v2.h hip.h
  HINTS ${CUBLAS_HINTS}
  PATH_SUFFIXES include inc include/x86_64 include/x64
  PATHS ${CUBLAS_PATHS}
  DOC "cuBLAS include header cublas_v2.h"
)
mark_as_advanced(CUBLAS_INCLUDE_DIRS)

# Finds the libraries
find_library(HIP_LIBRARIES
  NAMES hiprt
  HINTS ${CUBLAS_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CUBLAS_PATHS}
  DOC "HIP library"
)
mark_as_advanced(HIP_LIBRARIES)
find_library(CUBLAS_LIBRARIES
  NAMES cublas
  HINTS ${CUBLAS_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
  PATHS ${CUBLAS_PATHS}
  DOC "cuBLAS library"
)
mark_as_advanced(CUBLAS_LIBRARIES)

# =============================================================================

# Notification messages
if(NOT CUBLAS_INCLUDE_DIRS)
  message(STATUS "Could NOT find 'cuBLAS.h', install HIP/cuBLAS or set HIP_ROOT")
endif()
if(NOT HIP_LIBRARIES)
  message(STATUS "Could NOT find HIP library, install it or set HIP_ROOT")
endif()
if(NOT CUBLAS_LIBRARIES)
  message(STATUS "Could NOT find cuBLAS library, install it or set HIP_ROOT")
endif()

# Determines whether or not cuBLAS was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuBLAS DEFAULT_MSG CUBLAS_INCLUDE_DIRS HIP_LIBRARIES CUBLAS_LIBRARIES)

# =============================================================================
