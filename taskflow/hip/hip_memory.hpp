#pragma once

#include "hip_device.hpp"

/**
@file hip_memory.hpp
@brief hip memory utilities include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// memory
// ----------------------------------------------------------------------------

/**
@brief queries the free memory (expensive call)
*/
inline size_t hip_get_free_mem(int d) {
  hipScopedDevice ctx(d);
  size_t free, total;
  TF_CHECK_HIP(
    hipMemGetInfo(&free, &total), "failed to get mem info on device ", d
  );
  return free;
}

/**
@brief queries the total available memory (expensive call)
*/
inline size_t hip_get_total_mem(int d) {
  hipScopedDevice ctx(d);
  size_t free, total;
  TF_CHECK_HIP(
    hipMemGetInfo(&free, &total), "failed to get mem info on device ", d
  );
  return total;
}

/**
@brief allocates memory on the given device for holding @c N elements of type @c T

The function calls @c hipMalloc to allocate <tt>N*sizeof(T)</tt> bytes of memory
on the given device @c d and returns a pointer to the starting address of
the device memory.
*/
template <typename T>
T* hip_malloc_device(size_t N, int d) {
  hipScopedDevice ctx(d);
  T* ptr {nullptr};
  TF_CHECK_HIP(
    hipMalloc(&ptr, N*sizeof(T)),
    "failed to allocate memory (", N*sizeof(T), "bytes) on device ", d
  )
  return ptr;
}

/**
@brief allocates memory on the current device associated with the caller

The function calls malloc_device from the current device associated
with the caller.
*/
template <typename T>
T* hip_malloc_device(size_t N) {
  T* ptr {nullptr};
  TF_CHECK_HIP(
    hipMalloc(&ptr, N*sizeof(T)), 
    "failed to allocate memory (", N*sizeof(T), "bytes)"
  )
  return ptr;
}

/**
@brief allocates shared memory for holding @c N elements of type @c T

The function calls @c hipMallocManaged to allocate <tt>N*sizeof(T)</tt> bytes
of memory and returns a pointer to the starting address of the shared memory.
*/
template <typename T>
T* hip_malloc_shared(size_t N) {
  T* ptr {nullptr};
  TF_CHECK_HIP(
    hipMallocManaged(&ptr, N*sizeof(T)),
    "failed to allocate shared memory (", N*sizeof(T), "bytes)"
  )
  return ptr;
}

/**
@brief frees memory on the GPU device

@tparam T pointer type
@param ptr device pointer to memory to free
@param d device context identifier

This methods call @c hipFree to free the memory space pointed to by @c ptr
using the given device context.
*/
template <typename T>
void hip_free(T* ptr, int d) {
  hipScopedDevice ctx(d);
  TF_CHECK_HIP(hipFree(ptr), "failed to free memory ", ptr, " on GPU ", d);
}

/**
@brief frees memory on the GPU device

@tparam T pointer type
@param ptr device pointer to memory to free

This methods call @c hipFree to free the memory space pointed to by @c ptr
using the current device context of the caller.
*/
template <typename T>
void hip_free(T* ptr) {
  TF_CHECK_HIP(hipFree(ptr), "failed to free memory ", ptr);
}

/**
@brief copies data between host and device asynchronously through a stream

@param stream stream identifier
@param dst destination memory address
@param src source memory address
@param count size in bytes to copy

The method calls @c hipMemcpyAsync with the given @c stream
using @c hipMemcpyDefault to infer the memory space of the source and
the destination pointers. The memory areas may not overlap.
*/
inline void hip_memcpy_async(
  hipStream_t stream, void* dst, const void* src, size_t count
) {
  TF_CHECK_HIP(
    hipMemcpyAsync(dst, src, count, hipMemcpyDefault, stream),
    "failed to perform hipMemcpyAsync"
  );
}

/**
@brief initializes or sets GPU memory to the given value byte by byte

@param stream stream identifier
@param devPtr pointer to GPU mempry
@param value value to set for each byte of the specified memory
@param count size in bytes to set

The method calls @c hipMemsetAsync with the given @c stream
to fill the first @c count bytes of the memory area pointed to by @c devPtr
with the constant byte value @c value.
*/
inline void hip_memset_async(
  hipStream_t stream, void* devPtr, int value, size_t count
){
  TF_CHECK_HIP(
    hipMemsetAsync(devPtr, value, count, stream),
    "failed to perform hipMemsetAsync"
  );
}

// ----------------------------------------------------------------------------
// Shared Memory
// ----------------------------------------------------------------------------
//
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      extern __shared__  T sdata[];
//      ...
//      doStuff(sdata);
//      ...
//   }
//
//  With this:
//
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      hipSharedMemory<T> smem;
//      T* sdata = smem.get();
//      ...
//      doStuff(sdata);
//      ...
//   }
// ----------------------------------------------------------------------------

// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
/**
@private
*/
template <typename T>
struct hipSharedMemory
{
  // Ensure that we won't compile any un-specialized types
  __device__ T *get()
  {
    extern __device__ void error(void);
    error();
    return NULL;
  }
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

/**
@private
*/
template <>
struct hipSharedMemory <int>
{
  __device__ int *get()
  {
    extern __shared__ int s_int[];
    return s_int;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <unsigned int>
{
  __device__ unsigned int *get()
  {
    extern __shared__ unsigned int s_uint[];
    return s_uint;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <char>
{
  __device__ char *get()
  {
    extern __shared__ char s_char[];
    return s_char;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <unsigned char>
{
  __device__ unsigned char *get()
  {
    extern __shared__ unsigned char s_uchar[];
    return s_uchar;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <short>
{
  __device__ short *get()
  {
    extern __shared__ short s_short[];
    return s_short;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <unsigned short>
{
  __device__ unsigned short *get()
  {
    extern __shared__ unsigned short s_ushort[];
    return s_ushort;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <long>
{
  __device__ long *get()
  {
    extern __shared__ long s_long[];
    return s_long;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <unsigned long>
{
  __device__ unsigned long *get()
  {
    extern __shared__ unsigned long s_ulong[];
    return s_ulong;
  }
};

//template <>
//struct hipSharedMemory <size_t>
//{
//  __device__ size_t *get()
//  {
//    extern __shared__ size_t s_sizet[];
//    return s_sizet;
//  }
//};

/**
@private
*/
template <>
struct hipSharedMemory <bool>
{
  __device__ bool *get()
  {
    extern __shared__ bool s_bool[];
    return s_bool;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <float>
{
  __device__ float *get()
  {
    extern __shared__ float s_float[];
    return s_float;
  }
};

/**
@private
*/
template <>
struct hipSharedMemory <double>
{
  __device__ double *get()
  {
    extern __shared__ double s_double[];
    return s_double;
  }
};



// ----------------------------------------------------------------------------
// hipDeviceAllocator
// ----------------------------------------------------------------------------

/**
@class hipDeviceAllocator

@brief class to create a hip device allocator 

@tparam T element type

A %hipDeviceAllocator enables device-specific allocation for 
standard library containers. It is typically passed as template parameter 
when declaring standard library containers (e.g. std::vector).
*/
template<typename T>
class hipDeviceAllocator {

  public:

  /**
  @brief element type
  */
  using value_type = T;

  /**
  @brief element pointer type
  */
  using pointer = T*;

  /**
  @brief element reference type
  */
  using reference = T&;

  /**
  @brief const element pointer type
  */
  using const_pointer = const T*;

  /**
  @brief constant element reference type
  */
  using const_reference = const T&;

  /**
  @brief size type
  */
  using size_type = std::size_t;
  
  /**
  @brief pointer difference type
  */
  using difference_type = std::ptrdiff_t;

  /**
  @brief its member type @c U is the equivalent allocator type to allocate elements of type U
  */
  template<typename U> 
  struct rebind { 
    /**
    @brief allocator of a different data type
    */
    using other = hipDeviceAllocator<U>; 
  }; 

  /** 
  @brief Constructs a device allocator object.
  */
  hipDeviceAllocator() noexcept {}

  /**
  @brief Constructs a device allocator object from another device allocator object.
  */
  hipDeviceAllocator( const hipDeviceAllocator& ) noexcept {}

  /**
  @brief Constructs a device allocator object from another device allocator 
         object with a different element type.
  */
  template<typename U>
  hipDeviceAllocator( const hipDeviceAllocator<U>& ) noexcept {}

  /**
  @brief Destructs the device allocator object.
  */
  ~hipDeviceAllocator() noexcept {}

  /**
  @brief Returns the address of x.
  
  This effectively means returning &x.
  
  @param x reference to an object
  @return a pointer to the object
  */
  pointer address( reference x ) { return &x; }

  /**
  @brief Returns the address of x.
  
  This effectively means returning &x.
  
  @param x reference to an object
  @return a pointer to the object
  */
  const_pointer address( const_reference x ) const { return &x; }

  /** 
  @brief allocates block of storage.
  
  Attempts to allocate a block of storage with a size large enough to contain 
  @c n elements of member type, @c value_type, and returns a pointer 
  to the first element.
  
  The storage is aligned appropriately for object of type @c value_type, 
  but they are not constructed.
  
  The block of storage is allocated using hipMalloc and throws std::bad_alloc 
  if it cannot allocate the total amount of storage requested.
  
  @param n number of elements (each of size sizeof(value_type)) to be allocated
  @return a pointer to the initial element in the block of storage.
  */
  pointer allocate( size_type n, std::allocator<void>::const_pointer = 0 )
  {
    void* ptr = NULL;
    TF_CHECK_HIP(
      hipMalloc( &ptr, n*sizeof(T) ),
      "failed to allocate ", n, " elements (", n*sizeof(T), "bytes)"
    )
    return static_cast<pointer>(ptr);
  }

  /** 
  @brief Releases a block of storage previously allocated with member allocate and not yet released
  
  The elements in the array are not destroyed by a call to this member function.
  
  @param ptr pointer to a block of storage previously allocated with allocate
  */
  void deallocate( pointer ptr, size_type )
  {
    if(ptr){
      hipFree(ptr);
    }
  }

  /**
  @brief returns the maximum number of elements that could potentially 
         be allocated by this allocator
  
  A call to member allocate with the value returned by this function 
  can still fail to allocate the requested storage.
  
  @return the nubmer of elements that might be allcoated as maximum 
          by a call to member allocate
  */
  size_type max_size() const noexcept { return static_cast<size_type>(-1); }

  /**
  @brief ignored to avoid de-referencing device pointer from the host
  */
  void construct( pointer, const_reference) { }

  /**
  @brief ignored to avoid de-referencing device pointer from the host
  */
  void destroy( pointer) { }
  
  /**
  @brief compares two allocator of different types using @c ==

  Device allocators of different types are always equal to each other
  because the storage allocated by the allocator @c a1 can be deallocated 
  through @c a2. 
  */
  template <typename U>
  bool operator == (const hipDeviceAllocator<U>&) const noexcept {
    return true;
  }
  
  /**
  @brief compares two allocator of different types using @c !=

  Device allocators of different types are always equal to each other
  because the storage allocated by the allocator @c a1 can be deallocated 
  through @c a2. 
  */
  template <typename U>
  bool operator != (const hipDeviceAllocator<U>&) const noexcept {
    return false;
  }

};

// ----------------------------------------------------------------------------
// hipUSMAllocator
// ----------------------------------------------------------------------------

/**
@class hipUSMAllocator

@brief class to create a unified shared memory (USM) allocator 

@tparam T element type

A %hipUSMAllocator enables using unified shared memory (USM) allocation for 
standard library containers. It is typically passed as template parameter 
when declaring standard library containers (e.g. std::vector).
*/
template<typename T>
class hipUSMAllocator {

  public:

  /**
  @brief element type
  */
  using value_type = T;

  /**
  @brief element pointer type
  */
  using pointer = T*;

  /**
  @brief element reference type
  */
  using reference = T&;

  /**
  @brief const element pointer type
  */
  using const_pointer = const T*;

  /**
  @brief constant element reference type
  */
  using const_reference = const T&;

  /**
  @brief size type
  */
  using size_type = std::size_t;
  
  /**
  @brief pointer difference type
  */
  using difference_type = std::ptrdiff_t;

  /**
  @brief its member type @c U is the equivalent allocator type to allocate elements of type U
  */
  template<typename U> 
  struct rebind { 
    /**
    @brief allocator of a different data type
    */
    using other = hipUSMAllocator<U>; 
  }; 

  /** 
  @brief Constructs a device allocator object.
  */
  hipUSMAllocator() noexcept {}

  /**
  @brief Constructs a device allocator object from another device allocator object.
  */
  hipUSMAllocator( const hipUSMAllocator& ) noexcept {}

  /**
  @brief Constructs a device allocator object from another device allocator 
         object with a different element type.
  */
  template<typename U>
  hipUSMAllocator( const hipUSMAllocator<U>& ) noexcept {}

  /**
  @brief Destructs the device allocator object.
  */
  ~hipUSMAllocator() noexcept {}

  /**
  @brief Returns the address of x.
  
  This effectively means returning &x.
  
  @param x reference to an object
  @return a pointer to the object
  */
  pointer address( reference x ) { return &x; }

  /**
  @brief Returns the address of x.
  
  This effectively means returning &x.
  
  @param x reference to an object
  @return a pointer to the object
  */
  const_pointer address( const_reference x ) const { return &x; }

  /** 
  @brief allocates block of storage.
  
  Attempts to allocate a block of storage with a size large enough to contain 
  @c n elements of member type, @c value_type, and returns a pointer 
  to the first element.
  
  The storage is aligned appropriately for object of type @c value_type, 
  but they are not constructed.
  
  The block of storage is allocated using hipMalloc and throws std::bad_alloc 
  if it cannot allocate the total amount of storage requested.
  
  @param n number of elements (each of size sizeof(value_type)) to be allocated
  @return a pointer to the initial element in the block of storage.
  */
  pointer allocate( size_type n, std::allocator<void>::const_pointer = 0 )
  {
    void* ptr {nullptr};
    TF_CHECK_HIP(
      hipMallocManaged( &ptr, n*sizeof(T) ),
      "failed to allocate ", n, " elements (", n*sizeof(T), "bytes)"
    )
    return static_cast<pointer>(ptr);
  }

  /** 
  @brief Releases a block of storage previously allocated with member allocate and not yet released
  
  The elements in the array are not destroyed by a call to this member function.
  
  @param ptr pointer to a block of storage previously allocated with allocate
  */
  void deallocate( pointer ptr, size_type )
  {
    if(ptr){
      hipFree(ptr);
    }
  }

  /**
  @brief returns the maximum number of elements that could potentially 
         be allocated by this allocator
  
  A call to member allocate with the value returned by this function 
  can still fail to allocate the requested storage.
  
  @return the nubmer of elements that might be allcoated as maximum 
          by a call to member allocate
  */
  size_type max_size() const noexcept { return static_cast<size_type>(-1); }

  /**
  @brief Constructs an element object on the location pointed by ptr.
  @param ptr pointer to a location with enough storage soace to contain 
             an element of type @c value_type

  @param val value to initialize the constructed element to
  */
  void construct( pointer ptr, const_reference val ) {
    new ((void*)ptr) value_type(val);
  }

  /**
  @brief destroys in-place the object pointed by @c ptr
  
  Notice that this does not deallocate the storage for the element but calls
  its destructor.

  @param ptr pointer to the object to be destroye
  */
  void destroy( pointer ptr ) {
    ptr->~value_type();
  }

  /**
  @brief compares two allocator of different types using @c ==

  USM allocators of different types are always equal to each other
  because the storage allocated by the allocator @c a1 can be deallocated 
  through @c a2. 
  */
  template <typename U>
  bool operator == (const hipUSMAllocator<U>&) const noexcept {
    return true;
  }
  
  /**
  @brief compares two allocator of different types using @c !=

  USM allocators of different types are always equal to each other
  because the storage allocated by the allocator @c a1 can be deallocated 
  through @c a2. 
  */
  template <typename U>
  bool operator != (const hipUSMAllocator<U>&) const noexcept {
    return false;
  }

};

// ----------------------------------------------------------------------------
// GPU vector object
// ----------------------------------------------------------------------------

//template <typename T>
//using hipDeviceVector = std::vector<NoInit<T>, hipDeviceAllocator<NoInit<T>>>;

//template <typename T>
//using hipUSMVector = std::vector<T, hipUSMAllocator<T>>;

/**
@private
*/
template <typename T>
class hipDeviceVector {
  
  public:

    hipDeviceVector() = default;

    hipDeviceVector(size_t N) : _N {N} {
      if(N) {
        TF_CHECK_HIP(
          hipMalloc(&_data, N*sizeof(T)),
          "failed to allocate device memory (", N*sizeof(T), " bytes)"
        );
      }
    }
    
    hipDeviceVector(hipDeviceVector&& rhs) : 
      _data{rhs._data}, _N {rhs._N} {
      rhs._data = nullptr;
      rhs._N    = 0;
    }

    ~hipDeviceVector() {
      if(_data) {
        hipFree(_data);
      }
    }

    hipDeviceVector& operator = (hipDeviceVector&& rhs) {
      if(_data) {
        hipFree(_data);
      }
      _data = rhs._data;
      _N    = rhs._N;
      rhs._data = nullptr;
      rhs._N    = 0;
      return *this;
    }

    size_t size() const { return _N; }

    T* data() { return _data; }
    const T* data() const { return _data; }
    
    hipDeviceVector(const hipDeviceVector&) = delete;
    hipDeviceVector& operator = (const hipDeviceVector&) = delete;

  private:

    T* _data  {nullptr};
    size_t _N {0};
}; 


}  // end of namespace tf -----------------------------------------------------






