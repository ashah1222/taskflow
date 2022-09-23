#pragma once

#include "hip_task.hpp"
#include "hip_optimizer.hpp"

/**
@file hip_capturer.hpp
@brief %hipFlow capturer include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// class definition: hipFlowCapturer
// ----------------------------------------------------------------------------

/**
@class hipFlowCapturer

@brief class to create a %hipFlow graph using stream capture

The usage of tf::hipFlowCapturer is similar to tf::hipFlow, except users can
call the method tf::hipFlowCapturer::on to capture a sequence of asynchronous
hip operations through the given stream.
The following example creates a hip graph that captures two kernel tasks,
@c task_1 and @c task_2, where @c task_1 runs before @c task_2.

@code{.cpp}
taskflow.emplace([](tf::hipFlowCapturer& capturer){

  // capture my_kernel_1 through the given stream managed by the capturer
  auto task_1 = capturer.on([&](hipStream_t stream){
    my_kernel_1<<<grid_1, block_1, shm_size_1, stream>>>(my_parameters_1);
  });

  // capture my_kernel_2 through the given stream managed by the capturer
  auto task_2 = capturer.on([&](hipStream_t stream){
    my_kernel_2<<<grid_2, block_2, shm_size_2, stream>>>(my_parameters_2);
  });

  task_1.precede(task_2);
});
@endcode

Similar to tf::hipFlow, a %hipFlowCapturer is a task (tf::Task)
created from tf::Taskflow
and will be run by @em one worker thread in the executor.
That is, the callable that describes a %hipFlowCapturer
will be executed sequentially.
Inside a %hipFlow capturer task, different GPU tasks (tf::hipTask) may run
in parallel depending on the selected optimization algorithm.
By default, we use tf::hipRoundRobinCapturing to transform a user-level
graph into a native hip graph.

Please refer to @ref GPUTaskinghipFlowCapturer for details.
*/
class hipFlowCapturer {

  friend class hipFlow;
  friend class Executor;

  // created by user
  struct External {
    hipGraph graph;
  };
  
  // created from executor
  struct Internal {
    Internal(Executor& e) : executor{e} {}
    Executor& executor;
  };
  
  // created from hipFlow
  struct Proxy {
  };

  using handle_t = std::variant<External, Internal, Proxy>;

  using Optimizer = std::variant<
    hipRoundRobinCapturing,
    hipSequentialCapturing,
    hipLinearCapturing
  >;

  public:

    /**
    @brief constrcts a standalone hipFlowCapturer

    A standalone %hipFlow capturer does not go through any taskflow and
    can be run by the caller thread using explicit offload methods
    (e.g., tf::hipFlow::offload).
    */
    hipFlowCapturer();

    /**
    @brief destructs the hipFlowCapturer
    */
    virtual ~hipFlowCapturer();

    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;

    /**
    @brief queries the number of tasks
    */
    size_t num_tasks() const;

    /**
    @brief clear this %hipFlow capturer
    */
    void clear();

    /**
    @brief dumps the capture graph into a DOT format through an
           output stream
    */
    void dump(std::ostream& os) const;

    /**
    @brief selects a different optimization algorithm

    @tparam OPT optimizer type
    @tparam ArgsT arguments types

    @param args arguments to forward to construct the optimizer

    @return a reference to the optimizer

    We currently supports the following optimization algorithms to capture
    a user-described %hipFlow:
      + tf::hipSequentialCapturing
      + tf::hipRoundRobinCapturing
      + tf::hipLinearCapturing

    By default, tf::hipFlowCapturer uses the round-robin optimization
    algorithm with four streams to transform a user-level graph into
    a native hip graph.
    */
    template <typename OPT, typename... ArgsT>
    OPT& make_optimizer(ArgsT&&... args);

    // ------------------------------------------------------------------------
    // basic methods
    // ------------------------------------------------------------------------

    /**
    @brief captures a sequential hip operations from the given callable

    @tparam C callable type constructible with @c std::function<void(hipStream_t)>
    @param callable a callable to capture hip operations with the stream

    This methods applies a stream created by the flow to capture
    a sequence of hip operations defined in the callable.
    */
    template <typename C, std::enable_if_t<
      std::is_invocable_r_v<void, C, hipStream_t>, void>* = nullptr
    >
    hipTask on(C&& callable);

    /**
    @brief updates a capture task to another sequential hip operations

    The method is similar to hipFlowCapturer::on but operates
    on an existing task.
    */
    template <typename C, std::enable_if_t<
      std::is_invocable_r_v<void, C, hipStream_t>, void>* = nullptr
    >
    void on(hipTask task, C&& callable);

    /**
    @brief captures a no-operation task

    @return a tf::hipTask handle

    An empty node performs no operation during execution,
    but can be used for transitive ordering.
    For example, a phased execution graph with 2 groups of @c n nodes
    with a barrier between them can be represented using an empty node
    and @c 2*n dependency edges,
    rather than no empty node and @c n^2 dependency edges.
    */
    hipTask noop();

    /**
    @brief updates a task to a no-operation task

    The method is similar to tf::hipFlowCapturer::noop but
    operates on an existing task.
    */
    void noop(hipTask task);

    /**
    @brief copies data between host and device asynchronously through a stream

    @param dst destination memory address
    @param src source memory address
    @param count size in bytes to copy

    The method captures a @c hipMemcpyAsync operation through an
    internal stream.
    */
    hipTask memcpy(void* dst, const void* src, size_t count);

    /**
    @brief updates a capture task to a memcpy operation

    The method is similar to hipFlowCapturer::memcpy but operates on an
    existing task.
    */
    void memcpy(hipTask task, void* dst, const void* src, size_t count);

    /**
    @brief captures a copy task of typed data

    @tparam T element type (non-void)

    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param num number of elements to copy

    @return hipTask handle

    A copy task transfers <tt>num*sizeof(T)</tt> bytes of data from a source location
    to a target location. Direction can be arbitrary among CPUs and GPUs.
    */
    template <typename T,
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    hipTask copy(T* tgt, const T* src, size_t num);

    /**
    @brief updates a capture task to a copy operation

    The method is similar to hipFlowCapturer::copy but operates on
    an existing task.
    */
    template <typename T,
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    void copy(hipTask task, T* tgt, const T* src, size_t num);

    /**
    @brief initializes or sets GPU memory to the given value byte by byte

    @param ptr pointer to GPU mempry
    @param v value to set for each byte of the specified memory
    @param n size in bytes to set

    The method captures a @c hipMemsetAsync operation through an
    internal stream to fill the first @c count bytes of the memory area
    pointed to by @c devPtr with the constant byte value @c value.
    */
    hipTask memset(void* ptr, int v, size_t n);

    /**
    @brief updates a capture task to a memset operation

    The method is similar to hipFlowCapturer::memset but operates on
    an existing task.
    */
    void memset(hipTask task, void* ptr, int value, size_t n);

    /**
    @brief captures a kernel

    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type

    @param g configured grid
    @param b configured block
    @param s configured shared memory size in bytes
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return hipTask handle
    */
    template <typename F, typename... ArgsT>
    hipTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT&&... args);

    /**
    @brief updates a capture task to a kernel operation

    The method is similar to hipFlowCapturer::kernel but operates on
    an existing task.
    */
    template <typename F, typename... ArgsT>
    void kernel(
      hipTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
    );

    // ------------------------------------------------------------------------
    // generic algorithms
    // ------------------------------------------------------------------------

    /**
    @brief capturers a kernel to runs the given callable with only one thread

    @tparam C callable type

    @param c callable to run by a single kernel thread
    */
    template <typename C>
    hipTask single_task(C c);

    /**
    @brief updates a capture task to a single-threaded kernel

    This method is similar to hipFlowCapturer::single_task but operates
    on an existing task.
    */
    template <typename C>
    void single_task(hipTask task, C c);

    /**
    @brief captures a kernel that applies a callable to each dereferenced element
           of the data array

    @tparam I iterator type
    @tparam C callable type

    @param first iterator to the beginning
    @param last iterator to the end
    @param callable a callable object to apply to the dereferenced iterator

    @return hipTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    for(auto itr = first; itr != last; i++) {
      callable(*itr);
    }
    @endcode
    */
    template <typename I, typename C>
    hipTask for_each(I first, I last, C callable);

    /**
    @brief updates a capture task to a for-each kernel task

    This method is similar to hipFlowCapturer::for_each but operates
    on an existing task.
    */
    template <typename I, typename C>
    void for_each(hipTask task, I first, I last, C callable);

    /**
    @brief captures a kernel that applies a callable to each index in the range
           with the step size

    @tparam I index type
    @tparam C callable type

    @param first beginning index
    @param last last index
    @param step step size
    @param callable the callable to apply to each element in the data array

    @return hipTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    // step is positive [first, last)
    for(auto i=first; i<last; i+=step) {
      callable(i);
    }

    // step is negative [first, last)
    for(auto i=first; i>last; i+=step) {
      callable(i);
    }
    @endcode
    */
    template <typename I, typename C>
    hipTask for_each_index(I first, I last, I step, C callable);

    /**
    @brief updates a capture task to a for-each-index kernel task

    This method is similar to hipFlowCapturer::for_each_index but operates
    on an existing task.
    */
    template <typename I, typename C>
    void for_each_index(
      hipTask task, I first, I last, I step, C callable
    );

    /**
    @brief captures a kernel that transforms an input range to an output range

    @tparam I input iterator type
    @tparam O output iterator type
    @tparam C unary operator type

    @param first iterator to the beginning of the input range
    @param last iterator to the end of the input range
    @param output iterator to the beginning of the output range
    @param op unary operator to apply to transform each item in the range

    @return hipTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    while (first != last) {
      *output++ = op(*first++);
    }
    @endcode
    */
    template <typename I, typename O, typename C>
    hipTask transform(I first, I last, O output, C op);

    /**
    @brief updates a capture task to a transform kernel task

    This method is similar to hipFlowCapturer::transform but operates
    on an existing task.
    */
    template <typename I, typename O, typename C>
    void transform(hipTask task, I first, I last, O output, C op);

    /**
    @brief captures a kernel that transforms two input ranges to an output range

    @tparam I1 first input iterator type
    @tparam I2 second input iterator type
    @tparam O output iterator type
    @tparam C unary operator type

    @param first1 iterator to the beginning of the input range
    @param last1 iterator to the end of the input range
    @param first2 iterato
    @param output iterator to the beginning of the output range
    @param op binary operator to apply to transform each pair of items in the
              two input ranges

    @return hipTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    while (first1 != last1) {
      *output++ = op(*first1++, *first2++);
    }
    @endcode
    */
    template <typename I1, typename I2, typename O, typename C>
    hipTask transform(I1 first1, I1 last1, I2 first2, O output, C op);

    /**
    @brief updates a capture task to a transform kernel task

    This method is similar to hipFlowCapturer::transform but operates
    on an existing task.
    */
    template <typename I1, typename I2, typename O, typename C>
    void transform(
      hipTask task, I1 first1, I1 last1, I2 first2, O output, C op
    );

    /**
    @brief captures kernels that perform parallel reduction over a range of items

    @tparam I input iterator type
    @tparam T value type
    @tparam C binary operator type

    @param first iterator to the beginning
    @param last iterator to the end
    @param result pointer to the result with an initialized value
    @param op binary reduction operator

    @return a tf::hipTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    while (first != last) {
      *result = op(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename C>
    hipTask reduce(I first, I last, T* result, C op);

    /**
    @brief updates a capture task to a reduction task

    This method is similar to hipFlowCapturer::reduce but operates
    on an existing task.
    */
    template <typename I, typename T, typename C>
    void reduce(hipTask task, I first, I last, T* result, C op);

    /**
    @brief similar to tf::hipFlowCapturer::reduce but does not assume
           any initial value to reduce

    This method is equivalent to the parallel execution of the following loop
    on a GPU:

    @code{.cpp}
    *result = *first++;  // initial value does not involve in the loop
    while (first != last) {
      *result = op(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename C>
    hipTask uninitialized_reduce(I first, I last, T* result, C op);

    /**
    @brief updates a capture task to an uninitialized-reduction task

    This method is similar to hipFlowCapturer::uninitialized_reduce
    but operates on an existing task.
    */
    template <typename I, typename T, typename C>
    void uninitialized_reduce(
      hipTask task, I first, I last, T* result, C op
    );

    /**
    @brief captures kernels that perform parallel reduction over a range of
           transformed items

    @tparam I input iterator type
    @tparam T value type
    @tparam C binary operator type
    @tparam U unary operator type

    @param first iterator to the beginning
    @param last iterator to the end
    @param result pointer to the result with an initialized value
    @param bop binary reduce operator
    @param uop unary transform operator

    @return a tf::hipTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    while (first != last) {
      *result = bop(*result, uop(*first++));
    }
    @endcode
    */
    template <typename I, typename T, typename C, typename U>
    hipTask transform_reduce(I first, I last, T* result, C bop, U uop);

    /**
    @brief updates a capture task to a transform-reduce task

    This method is similar to hipFlowCapturer::transform_reduce but
    operates on an existing task.
    */
    template <typename I, typename T, typename C, typename U>
    void transform_reduce(
      hipTask task, I first, I last, T* result, C bop, U uop
    );

    /**
    @brief similar to tf::hipFlowCapturer::transform_reduce but does not assume
           any initial value to reduce

    This method is equivalent to the parallel execution of the following loop
    on a GPU:

    @code{.cpp}
    *result = uop(*first++);  // initial value does not involve in the loop
    while (first != last) {
      *result = bop(*result, uop(*first++));
    }
    @endcode
    */
    template <typename I, typename T, typename C, typename U>
    hipTask transform_uninitialized_reduce(I first, I last, T* result, C bop, U uop);

    /**
    @brief updates a capture task to a transform-reduce task of no initialized value

    This method is similar to hipFlowCapturer::transform_uninitialized_reduce
    but operates on an existing task.
    */
    template <typename I, typename T, typename C, typename U>
    void transform_uninitialized_reduce(
      hipTask task, I first, I last, T* result, C bop, U uop
    );

    /**
    @brief captures kernels that perform parallel inclusive scan
           over a range of items

    @tparam I input iterator type
    @tparam O output iterator type
    @tparam C binary operator type

    @param first iterator to the beginning
    @param last iterator to the end
    @param output iterator to the beginning of the output
    @param op binary operator

    @return a tf::hipTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    for(size_t i=0; i<std::distance(first, last); i++) {
      *(output + i) = i ? op(*(first+i), *(output+i-1)) : *(first+i);
    }
    @endcode
    */
    template <typename I, typename O, typename C>
    hipTask inclusive_scan(I first, I last, O output, C op);

    /**
    @brief updates a capture task to an inclusive scan task

    This method is similar to hipFlowCapturer::inclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename C>
    void inclusive_scan(hipTask task, I first, I last, O output, C op);

    /**
    @brief similar to hipFlowCapturer::inclusive_scan
           but excludes the first value
    */
    template <typename I, typename O, typename C>
    hipTask exclusive_scan(I first, I last, O output, C op);

    /**
    @brief updates a capture task to an exclusive scan task

    This method is similar to hipFlowCapturer::exclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename C>
    void exclusive_scan(hipTask task, I first, I last, O output, C op);

    /**
    @brief captures kernels that perform parallel inclusive scan
           over a range of transformed items

    @tparam I input iterator type
    @tparam O output iterator type
    @tparam B binary operator type
    @tparam U unary operator type

    @param first iterator to the beginning
    @param last iterator to the end
    @param output iterator to the beginning of the output
    @param bop binary operator
    @param uop unary operator

    @return a tf::hipTask handle

    This method is equivalent to the parallel execution of the following loop
    on a GPU:

    @code{.cpp}
    for(size_t i=0; i<std::distance(first, last); i++) {
      *(output + i) = i ? op(uop(*(first+i)), *(output+i-1)) : uop(*(first+i));
    }
    @endcode
     */
    template <typename I, typename O, typename B, typename U>
    hipTask transform_inclusive_scan(I first, I last, O output, B bop, U uop);

    /**
    @brief updates a capture task to a transform-inclusive scan task

    This method is similar to hipFlowCapturer::transform_inclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename B, typename U>
    void transform_inclusive_scan(
      hipTask task, I first, I last, O output, B bop, U uop
    );

    /**
    @brief similar to hipFlowCapturer::transform_inclusive_scan but
           excludes the first value
    */
    template <typename I, typename O, typename B, typename U>
    hipTask transform_exclusive_scan(I first, I last, O output, B bop, U uop);

    /**
    @brief updates a capture task to a transform-exclusive scan task

    This method is similar to hipFlowCapturer::transform_exclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename B, typename U>
    void transform_exclusive_scan(
      hipTask task, I first, I last, O output, B bop, U uop
    );

    /**
    @brief captures kernels that perform parallel merge on two sorted arrays

    @tparam A iterator type of the first input array
    @tparam B iterator type of the second input array
    @tparam C iterator type of the output array
    @tparam Comp comparator type

    @param a_first iterator to the beginning of the first input array
    @param a_last iterator to the end of the first input array
    @param b_first iterator to the beginning of the second input array
    @param b_last iterator to the end of the second input array
    @param c_first iterator to the beginning of the output array
    @param comp binary comparator

    @return a tf::hipTask handle

    Merges two sorted ranges <tt>[a_first, a_last)</tt> and
    <tt>[b_first, b_last)</tt> into one sorted range beginning at @c c_first.

    A sequence is said to be sorted with respect to a comparator @c comp
    if for any iterator it pointing to the sequence and
    any non-negative integer @c n such that <tt>it + n</tt> is a valid iterator
    pointing to an element of the sequence, <tt>comp(*(it + n), *it)</tt>
    evaluates to @c false.
     */
    template <typename A, typename B, typename C, typename Comp>
    hipTask merge(A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp);

    /**
    @brief updates a capture task to a merge task

    This method is similar to hipFlowCapturer::merge but operates
    on an existing task.
     */
    template <typename A, typename B, typename C, typename Comp>
    void merge(
      hipTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
    );

    /**
    @brief captures kernels that perform parallel key-value merge

    @tparam a_keys_it first key iterator type
    @tparam a_vals_it first value iterator type
    @tparam b_keys_it second key iterator type
    @tparam b_vals_it second value iterator type
    @tparam c_keys_it output key iterator type
    @tparam c_vals_it output value iterator type
    @tparam C comparator type

    @param a_keys_first iterator to the beginning of the first key range
    @param a_keys_last iterator to the end of the first key range
    @param a_vals_first iterator to the beginning of the first value range
    @param b_keys_first iterator to the beginning of the second key range
    @param b_keys_last iterator to the end of the second key range
    @param b_vals_first iterator to the beginning of the second value range
    @param c_keys_first iterator to the beginning of the output key range
    @param c_vals_first iterator to the beginning of the output value range
    @param comp comparator

    Performs a key-value merge that copies elements from
    <tt>[a_keys_first, a_keys_last)</tt> and <tt>[b_keys_first, b_keys_last)</tt>
    into a single range, <tt>[c_keys_first, c_keys_last + (a_keys_last - a_keys_first) + (b_keys_last - b_keys_first))</tt>
    such that the resulting range is in ascending key order.

    At the same time, the merge copies elements from the two associated ranges
    <tt>[a_vals_first + (a_keys_last - a_keys_first))</tt> and
    <tt>[b_vals_first + (b_keys_last - b_keys_first))</tt> into a single range,
    <tt>[c_vals_first, c_vals_first + (a_keys_last - a_keys_first) + (b_keys_last - b_keys_first))</tt>
    such that the resulting range is in ascending order
    implied by each input element's associated key.

    For example, assume:
      + @c a_keys = <tt>{8, 1}</tt>
      + @c a_vals = <tt>{1, 2}</tt>
      + @c b_keys = <tt>{3, 7}</tt>
      + @c b_vals = <tt>{3, 4}</tt>

    After the merge, we have:
      + @c c_keys = <tt>{1, 3, 7, 8}</tt>
      + @c c_vals = <tt>{2, 3, 4, 1}</tt>
    */
    template<
      typename a_keys_it, typename a_vals_it,
      typename b_keys_it, typename b_vals_it,
      typename c_keys_it, typename c_vals_it,
      typename C
    >
    hipTask merge_by_key(
      a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
      b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
      c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
    );

    /**
    @brief updates a capture task to a key-value merge task

    This method is similar to tf::hipFlowCapturer::merge_by_key but operates
    on an existing task.
    */
    template<
      typename a_keys_it, typename a_vals_it,
      typename b_keys_it, typename b_vals_it,
      typename c_keys_it, typename c_vals_it,
      typename C
    >
    void merge_by_key(
      hipTask task,
      a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
      b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
      c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
    );

    /**
    @brief captures kernels that sort the given array

    @tparam I iterator type of the first input array
    @tparam C comparator type

    @param first iterator to the beginning of the input array
    @param last iterator to the end of the input array
    @param comp binary comparator

    @return a tf::hipTask handle

    Sorts elements in the range <tt>[first, last)</tt>
    with the given comparator.
    */
    template <typename I, typename C>
    hipTask sort(I first, I last, C comp);

    /**
    @brief updates a capture task to a sort task

    This method is similar to hipFlowCapturer::sort but operates on
    an existing task.
    */
    template <typename I, typename C>
    void sort(hipTask task, I first, I last, C comp);

    /**
    @brief captures kernels that sort the given array

    @tparam K_it iterator type of the key
    @tparam V_it iterator type of the value
    @tparam C comparator type

    @param k_first iterator to the beginning of the key array
    @param k_last iterator to the end of the key array
    @param v_first iterator to the beginning of the value array
    @param comp binary comparator

    @return a tf::hipTask handle

    Sorts key-value elements in <tt>[k_first, k_last)</tt> and
    <tt>[v_first, v_first + (k_last - k_first))</tt> into ascending key order
    using the given comparator @c comp.
    If @c i and @c j are any two valid iterators in <tt>[k_first, k_last)</tt>
    such that @c i precedes @c j, and @c p and @c q are iterators in
    <tt>[v_first, v_first + (k_last - k_first))</tt> corresponding to
    @c i and @c j respectively, then <tt>comp(*j, *i)</tt> evaluates to @c false.

    For example, assume:
      + @c keys are <tt>{1, 4, 2, 8, 5, 7}</tt>
      + @c values are <tt>{'a', 'b', 'c', 'd', 'e', 'f'}</tt>

    After sort:
      + @c keys are <tt>{1, 2, 4, 5, 7, 8}</tt>
      + @c values are <tt>{'a', 'c', 'b', 'e', 'f', 'd'}</tt>
    */
    template <typename K_it, typename V_it, typename C>
    hipTask sort_by_key(K_it k_first, K_it k_last, V_it v_first, C comp);

    /**
    @brief updates a capture task to a key-value sort task

    This method is similar to tf::hipFlowCapturer::sort_by_key
    but operates on an existing task.
    */
    template <typename K_it, typename V_it, typename C>
    void sort_by_key(
      hipTask task, K_it k_first, K_it k_last, V_it v_first, C comp
    );

    /**
    @brief creates a task to find the index of the first element in a range

    @tparam I input iterator type
    @tparam U unary operator type

    @param first iterator to the beginning of the range
    @param last iterator to the end of the range
    @param idx pointer to the index of the found element
    @param op unary operator which returns @c true for the required element

    Finds the index @c idx of the first element in the range
    <tt>[first, last)</tt> such that <tt>op(*(first+idx))</tt> is true.
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
    template <typename I, typename U>
    hipTask find_if(I first, I last, unsigned* idx, U op);

    /**
    @brief updates the parameters of a find-if task

    This method is similar to tf::hipFlowCapturer::find_if but operates
    on an existing task.
    */
    template <typename I, typename U>
    void find_if(hipTask task, I first, I last, unsigned* idx, U op);

    /**
    @brief finds the index of the minimum element in a range

    @tparam I input iterator type
    @tparam O comparator type

    @param first iterator to the beginning of the range
    @param last iterator to the end of the range
    @param idx solution index of the minimum element
    @param op comparison function object

    The function launches kernels asynchronously to find
    the smallest element in the range <tt>[first, last)</tt>
    using the given comparator @c op.
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
    template <typename I, typename O>
    hipTask min_element(I first, I last, unsigned* idx, O op);

    /**
    @brief updates the parameters of a min-element task

    This method is similar to hipFlowCapturer::min_element but operates
    on an existing task.
    */
    template <typename I, typename O>
    void min_element(hipTask task, I first, I last, unsigned* idx, O op);

    /**
    @brief finds the index of the maximum element in a range

    @tparam I input iterator type
    @tparam O comparator type

    @param first iterator to the beginning of the range
    @param last iterator to the end of the range
    @param idx solution index of the maximum element
    @param op comparison function object

    The function launches kernels asynchronously to find
    the largest element in the range <tt>[first, last)</tt>
    using the given comparator @c op.
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
    template <typename I, typename O>
    hipTask max_element(I first, I last, unsigned* idx, O op);

    /**
    @brief updates the parameters of a max-element task

    This method is similar to hipFlowCapturer::max_element but operates
    on an existing task.
     */
    template <typename I, typename O>
    void max_element(hipTask task, I first, I last, unsigned* idx, O op);

    // ------------------------------------------------------------------------
    // offload methods
    // ------------------------------------------------------------------------

    /**
    @brief offloads the captured %hipFlow onto a GPU and repeatedly runs it until
    the predicate becomes true

    @tparam P predicate type (a binary callable)

    @param predicate a binary predicate (returns @c true for stop)

    Immediately offloads the %hipFlow captured so far onto a GPU and
    repeatedly runs it until the predicate returns @c true.

    By default, if users do not offload the %hipFlow capturer,
    the executor will offload it once.
    */
    template <typename P>
    void offload_until(P&& predicate);

    /**
    @brief offloads the captured %hipFlow and executes it by the given times

    @param n number of executions
    */
    void offload_n(size_t n);

    /**
    @brief offloads the captured %hipFlow and executes it once
    */
    void offload();

  private:

    handle_t _handle;

    hipGraph& _graph;

    Optimizer _optimizer;

    hipGraphExec _exec {nullptr};

    hipFlowCapturer(hipGraph&, Executor& executor);
    hipFlowCapturer(hipGraph&);

    hipGraph_t _capture();
};

// constructs a hipFlow capturer from a taskflow
inline hipFlowCapturer::hipFlowCapturer(hipGraph& g) :
  _handle {std::in_place_type_t<Proxy>{}},
  _graph  {g} {
}

// constructs a hipFlow capturer from a taskflow
inline hipFlowCapturer::hipFlowCapturer(hipGraph& g, Executor& e) :
  _handle {std::in_place_type_t<Internal>{}, e},
  _graph  {g} {
}

// constructs a standalone hipFlow capturer
inline hipFlowCapturer::hipFlowCapturer() :
  _handle {std::in_place_type_t<External>{}},
  _graph  {std::get_if<External>(&_handle)->graph} {
}

inline hipFlowCapturer::~hipFlowCapturer() {
}

// Function: empty
inline bool hipFlowCapturer::empty() const {
  return _graph.empty();
}

// Function: num_tasks
inline size_t hipFlowCapturer::num_tasks() const {
  return _graph._nodes.size();
}

// Procedure: clear
inline void hipFlowCapturer::clear() {
  _exec.clear();
  _graph._nodes.clear();
}

// Procedure: dump
inline void hipFlowCapturer::dump(std::ostream& os) const {
  _graph.dump(os, nullptr, "");
}

// Function: capture
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, hipStream_t>, void>*
>
hipTask hipFlowCapturer::on(C&& callable) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<hipNode::Capture>{}, std::forward<C>(callable)
  );
  return hipTask(node);
}

// Function: noop
inline hipTask hipFlowCapturer::noop() {
  return on([](hipStream_t){});
}

// Function: noop
inline void hipFlowCapturer::noop(hipTask task) {
  on(task, [](hipStream_t){});
}

// Function: memcpy
inline hipTask hipFlowCapturer::memcpy(
  void* dst, const void* src, size_t count
) {
  return on([dst, src, count] (hipStream_t stream) mutable {
    TF_CHECK_HIP(
      hipMemcpyAsync(dst, src, count, hipMemcpyDefault, stream),
      "failed to capture memcpy"
    );
  });
}

// Function: copy
template <typename T, std::enable_if_t<!std::is_same_v<T, void>, void>*>
hipTask hipFlowCapturer::copy(T* tgt, const T* src, size_t num) {
  return on([tgt, src, num] (hipStream_t stream) mutable {
    TF_CHECK_HIP(
      hipMemcpyAsync(tgt, src, sizeof(T)*num, hipMemcpyDefault, stream),
      "failed to capture copy"
    );
  });
}

// Function: memset
inline hipTask hipFlowCapturer::memset(void* ptr, int v, size_t n) {
  return on([ptr, v, n] (hipStream_t stream) mutable {
    TF_CHECK_HIP(
      hipMemsetAsync(ptr, v, n, stream), "failed to capture memset"
    );
  });
}

// Function: kernel
template <typename F, typename... ArgsT>
hipTask hipFlowCapturer::kernel(
  dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
) {
  return on([g, b, s, f, args...] (hipStream_t stream) mutable {
    f<<<g, b, s, stream>>>(args...);
  });
}

// Function: _capture
inline hipGraph_t hipFlowCapturer::_capture() {
  return std::visit(
    [this](auto&& opt){ return opt._optimize(_graph); }, _optimizer
  );
}

// Procedure: offload_until
template <typename P>
void hipFlowCapturer::offload_until(P&& predicate) {

  // If the topology got changed, we need to destroy the executable
  // and create a new one
  if(_graph._state & hipGraph::CHANGED) {
    // TODO: store the native graph?
    hipGraphNative g(_capture());
    _exec.instantiate(g);
  }
  // if the graph is just updated (i.e., topology does not change),
  // we can skip part of the optimization and just update the executable
  // with the new captured graph
  else if(_graph._state & hipGraph::UPDATED) {
    // TODO: skip part of the optimization (e.g., levelization)
    hipGraphNative g(_capture());
    if(_exec.update(g) != hipGraphExecUpdateSuccess) {
      _exec.instantiate(g);
    }
    // TODO: store the native graph?
  }

  // offload the executable
  if(_exec) {
    hipStream s;
    while(!predicate()) {
      _exec.launch(s);
      s.synchronize();
    }
  }

  _graph._state = hipGraph::OFFLOADED;
}

// Procedure: offload_n
inline void hipFlowCapturer::offload_n(size_t n) {
  offload_until([repeat=n] () mutable { return repeat-- == 0; });
}

// Procedure: offload
inline void hipFlowCapturer::offload() {
  offload_until([repeat=1] () mutable { return repeat-- == 0; });
}

// Function: on
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, hipStream_t>, void>*
>
void hipFlowCapturer::on(hipTask task, C&& callable) {

  if(task.type() != hipTaskType::CAPTURE) {
    TF_THROW("invalid hipTask type (must be CAPTURE)");
  }

  _graph._state |= hipGraph::UPDATED;

  std::get_if<hipNode::Capture>(&task._node->_handle)->work =
    std::forward<C>(callable);
}

// Function: memcpy
inline void hipFlowCapturer::memcpy(
  hipTask task, void* dst, const void* src, size_t count
) {
  on(task, [dst, src, count](hipStream_t stream) mutable {
    TF_CHECK_HIP(
      hipMemcpyAsync(dst, src, count, hipMemcpyDefault, stream),
      "failed to capture memcpy"
    );
  });
}

// Function: copy
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
void hipFlowCapturer::copy(
  hipTask task, T* tgt, const T* src, size_t num
) {
  on(task, [tgt, src, num] (hipStream_t stream) mutable {
    TF_CHECK_HIP(
      hipMemcpyAsync(tgt, src, sizeof(T)*num, hipMemcpyDefault, stream),
      "failed to capture copy"
    );
  });
}

// Function: memset
inline void hipFlowCapturer::memset(
  hipTask task, void* ptr, int v, size_t n
) {
  on(task, [ptr, v, n] (hipStream_t stream) mutable {
    TF_CHECK_HIP(
      hipMemsetAsync(ptr, v, n, stream), "failed to capture memset"
    );
  });
}

// Function: kernel
template <typename F, typename... ArgsT>
void hipFlowCapturer::kernel(
  hipTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
) {
  on(task, [g, b, s, f, args...] (hipStream_t stream) mutable {
    f<<<g, b, s, stream>>>(args...);
  });
}

// Function: make_optimizer
template <typename OPT, typename ...ArgsT>
OPT& hipFlowCapturer::make_optimizer(ArgsT&&... args) {
  return _optimizer.emplace<OPT>(std::forward<ArgsT>(args)...);
}

}  // end of namespace tf -----------------------------------------------------
