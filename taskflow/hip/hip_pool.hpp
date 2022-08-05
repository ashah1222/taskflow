#pragma once

#include "hip_error.hpp"

namespace tf {

/**
@brief per-thread object pool to manage hip device object

@tparam H object type
@tparam C function object to create a library object
@tparam D function object to delete a library object

A hip device object has a lifetime associated with a device,
for example, @c hipStream_t, @c cublasHandle_t, etc.
Creating a device object is typically expensive (e.g., 10-200 ms)
and destroying it may trigger implicit device synchronization.
For applications tha intensively make use of device objects,
it is desirable to reuse them as much as possible.

There exists an one-to-one relationship between hip devices in hip Runtime API
and CUcontexts in the hip Driver API within a process.
The specific context which the hip Runtime API uses for a device
is called the device's primary context.
From the perspective of the hip Runtime API,
a device and its primary context are synonymous.

We design the device object pool in a decentralized fashion by keeping
(1) a global pool to keep track of potentially usable objects and
(2) a per-thread pool to footprint objects with shared ownership.
The global pool does not own the object and therefore does not destruct any of them.
The per-thread pool keeps the footprints of objects with shared ownership
and will destruct them if the thread holds the last reference count after it joins.
The motivation of this decentralized control is to avoid device objects
from being destroyed while the context had been destroyed due to driver shutdown.

*/
template <typename H, typename C, typename D>
class hipPerThreadDeviceObjectPool {

  public:

  /**
  @brief structure to store a context object
   */
  struct Object {

    int device;
    H value;

    Object(int);
    ~Object();

    Object(const Object&) = delete;
    Object(Object&&) = delete;
  };

  private:

  // Master thread hold the storage to the pool.
  // Due to some ordering, hip context may be destroyed when the master
  // program thread destroys the hip object.
  // Therefore, we use a decentralized approach to let child thread
  // destroy hip objects while the master thread only keeps a weak reference
  // to those objects for reuse.
  struct hipGlobalDeviceObjectPool {

    std::shared_ptr<Object> acquire(int);
    void release(int, std::weak_ptr<Object>);

    std::mutex mutex;
    std::unordered_map<int, std::vector<std::weak_ptr<Object>>> pool;
  };

  public:

    /**
    @brief default constructor
     */
    hipPerThreadDeviceObjectPool() = default;

    /**
    @brief acquires a device object with shared ownership
     */
    std::shared_ptr<Object> acquire(int);

    /**
    @brief releases a device object with moved ownership
    */
    void release(std::shared_ptr<Object>&&);

    /**
    @brief queries the number of device objects with shared ownership
     */
    size_t footprint_size() const;

  private:

    inline static hipGlobalDeviceObjectPool _shared_pool;

    std::unordered_set<std::shared_ptr<Object>> _footprint;
};

// ----------------------------------------------------------------------------
// hipPerThreadDeviceObject::hipHanale definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
hipPerThreadDeviceObjectPool<H, C, D>::Object::Object(int d) :
  device {d} {
  hipScopedDevice ctx(device);
  value = C{}();
}

template <typename H, typename C, typename D>
hipPerThreadDeviceObjectPool<H, C, D>::Object::~Object() {
  hipScopedDevice ctx(device);
  D{}(value);
}

// ----------------------------------------------------------------------------
// hipPerThreadDeviceObject::hipHanaldePool definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename hipPerThreadDeviceObjectPool<H, C, D>::Object>
hipPerThreadDeviceObjectPool<H, C, D>::hipGlobalDeviceObjectPool::acquire(int d) {
  std::scoped_lock<std::mutex> lock(mutex);
  if(auto itr = pool.find(d); itr != pool.end()) {
    while(!itr->second.empty()) {
      auto sptr = itr->second.back().lock();
      itr->second.pop_back();
      if(sptr) {
        return sptr;
      }
    }
  }
  return nullptr;
}

template <typename H, typename C, typename D>
void hipPerThreadDeviceObjectPool<H, C, D>::hipGlobalDeviceObjectPool::release(
  int d, std::weak_ptr<Object> ptr
) {
  std::scoped_lock<std::mutex> lock(mutex);
  pool[d].push_back(ptr);
}

// ----------------------------------------------------------------------------
// hipPerThreadDeviceObject definition
// ----------------------------------------------------------------------------

template <typename H, typename C, typename D>
std::shared_ptr<typename hipPerThreadDeviceObjectPool<H, C, D>::Object>
hipPerThreadDeviceObjectPool<H, C, D>::acquire(int d) {

  auto ptr = _shared_pool.acquire(d);

  if(!ptr) {
    ptr = std::make_shared<Object>(d);
  }

  return ptr;
}

template <typename H, typename C, typename D>
void hipPerThreadDeviceObjectPool<H, C, D>::release(
  std::shared_ptr<Object>&& ptr
) {
  _shared_pool.release(ptr->device, ptr);
  _footprint.insert(std::move(ptr));
}

template <typename H, typename C, typename D>
size_t hipPerThreadDeviceObjectPool<H, C, D>::footprint_size() const {
  return _footprint.size();
}

}  // end of namespace tf -----------------------------------------------------



