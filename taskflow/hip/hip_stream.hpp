#pragma once

#include "hip_pool.hpp"

/**
@file hip_stream.hpp
@brief hip stream utilities include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// hipStream
// ----------------------------------------------------------------------------

/**
@class hipStream

@brief class to create an RAII-styled wrapper over a native hip stream

A hipStream object is an RAII-styled wrapper over a native hip stream
(@c hipStream_t).
A hipStream object is move-only.
*/
class hipStream {

  struct hipStreamCreator {
    hipStream_t operator () () const {
      hipStream_t stream;
      TF_CHECK_HIP(hipStreamCreate(&stream), "failed to create a hip stream");
      return stream;
    }
  };
  
  struct hipStreamDeleter {
    void operator () (hipStream_t stream) const {
      if(stream) {
        hipStreamDestroy(stream);
      }
    }
  };
  
  public:

    /**
    @brief constructs an RAII-styled object from the given hip stream

    Constructs a hipStream object which owns @c stream.
    */
    explicit hipStream(hipStream_t stream) : _stream(stream) {
    }
    
    /**
    @brief constructs an RAII-styled object for a new hip stream

    Equivalently calling @c hipStreamCreate to create a stream.
    */
    hipStream() : _stream{ hipStreamCreator{}() } {
    }
    
    /**
    @brief disabled copy constructor
    */
    hipStream(const hipStream&) = delete;
    
    /**
    @brief move constructor
    */
    hipStream(hipStream&& rhs) : _stream{rhs._stream} {
      rhs._stream = nullptr;
    }

    /**
    @brief destructs the hip stream
    */
    ~hipStream() {
      hipStreamDeleter {} (_stream);
    }
    
    /**
    @brief disabled copy assignment
    */
    hipStream& operator = (const hipStream&) = delete;

    /**
    @brief move assignment
    */
    hipStream& operator = (hipStream&& rhs) {
      hipStreamDeleter {} (_stream);
      _stream = rhs._stream;
      rhs._stream = nullptr;
      return *this;
    }
    
    /**
    @brief implicit conversion to the native hip stream (hipStream_t)

    Returns the underlying stream of type @c hipStream_t.
    */
    operator hipStream_t () const {
      return _stream;
    }
    
    /**
    @brief synchronizes the associated stream

    Equivalently calling @c hipStreamSynchronize to block 
    until this stream has completed all operations.
    */
    void synchronize() const {
      TF_CHECK_HIP(
        hipStreamSynchronize(_stream), "failed to synchronize a hip stream"
      );
    }
    
    /**
    @brief begins graph capturing on the stream

    When a stream is in capture mode, all operations pushed into the stream 
    will not be executed, but will instead be captured into a graph, 
    which will be returned via hipStream::end_capture. 

    A thread's mode can be one of the following:
    + @c hipStreamCaptureModeGlobal: This is the default mode. 
      If the local thread has an ongoing capture sequence that was not initiated 
      with @c hipStreamCaptureModeRelaxed at @c cuStreamBeginCapture, 
      or if any other thread has a concurrent capture sequence initiated with 
      @c hipStreamCaptureModeGlobal, this thread is prohibited from potentially 
      unsafe API calls.

    + @c hipStreamCaptureModeThreadLocal: If the local thread has an ongoing capture 
      sequence not initiated with @c hipStreamCaptureModeRelaxed, 
      it is prohibited from potentially unsafe API calls. 
      Concurrent capture sequences in other threads are ignored.

    + @c hipStreamCaptureModeRelaxed: The local thread is not prohibited 
      from potentially unsafe API calls. Note that the thread is still prohibited 
      from API calls which necessarily conflict with stream capture, for example, 
      attempting @c hipEventQuery on an event that was last recorded 
      inside a capture sequence.
    */
    void begin_capture(hipStreamCaptureMode m = hipStreamCaptureModeGlobal) const {
      TF_CHECK_HIP(
        hipStreamBeginCapture(_stream, m), 
        "failed to begin capture on stream ", _stream, " with thread mode ", m
      );
    }

    /**
    @brief ends graph capturing on the stream
    
    Equivalently calling @c hipStreamEndCapture to
    end capture on stream and returning the captured graph. 
    Capture must have been initiated on stream via a call to hipStream::begin_capture. 
    If capture was invalidated, due to a violation of the rules of stream capture, 
    then a NULL graph will be returned.
    */
    hipGraph_t end_capture() const {
      hipGraph_t native_g;
      TF_CHECK_HIP(
        hipStreamEndCapture(_stream, &native_g), 
        "failed to end capture on stream ", _stream
      );
      return native_g;
    }
    
    /**
    @brief records an event on the stream

    Equivalently calling @c hipEventRecord to record an event on this stream,
    both of which must be on the same hip context.
    */
    void record(hipEvent_t event) const {
      TF_CHECK_HIP(
        hipEventRecord(event, _stream), 
        "failed to record event ", event, " on stream ", _stream
      );
    }
    
    /**
    @brief waits on an event

    Equivalently calling @c hipStreamWaitEvent to make all future work 
    submitted to stream wait for all work captured in event.
    */
    void wait(hipEvent_t event) const {
      TF_CHECK_HIP(
        hipStreamWaitEvent(_stream, event, 0), 
        "failed to wait for event ", event, " on stream ", _stream
      );
    }

  private:

    hipStream_t _stream {nullptr};
};

// ----------------------------------------------------------------------------
// hipEvent
// ----------------------------------------------------------------------------

/**
@class hipEvent

@brief class to create an RAII-styled wrapper over a native hip event

A hipEvent object is an RAII-styled wrapper over a native hip event 
(@c hipEvent_t).
A hipEvent object is move-only.
*/
class hipEvent {

  struct hipEventCreator {
  
    hipEvent_t operator () () const {
      hipEvent_t event;
      TF_CHECK_HIP(hipEventCreate(&event), "failed to create a hip event");
      return event;
    }
    
    hipEvent_t operator () (unsigned int flag) const {
      hipEvent_t event;
      TF_CHECK_HIP(
        hipEventCreateWithFlags(&event, flag),
        "failed to create a hip event with flag=", flag
      );
      return event;
    }
  };
  
  struct hipEventDeleter {
    void operator () (hipEvent_t event) const {
      hipEventDestroy(event);
    }
  };

  public:

    /**
    @brief constructs an RAII-styled hip event object from the given hip event
    */
    explicit hipEvent(hipEvent_t event) : _event(event) { }   

    /**
    @brief constructs an RAII-styled hip event object
    */
    hipEvent() : _event{ hipEventCreator{}() } { }
    
    /**
    @brief constructs an RAII-styled hip event object with the given flag
    */
    explicit hipEvent(unsigned int flag) : _event{ hipEventCreator{}(flag) } { }
    
    /**
    @brief disabled copy constructor
    */
    hipEvent(const hipEvent&) = delete;
    
    /**
    @brief move constructor
    */
    hipEvent(hipEvent&& rhs) : _event{rhs._event} {
      rhs._event = nullptr;
    }

    /**
    @brief destructs the hip event
    */
    ~hipEvent() {
      hipEventDeleter {} (_event);
    }
    
    /**
    @brief disabled copy assignment
    */
    hipEvent& operator = (const hipEvent&) = delete;

    /**
    @brief move assignment
    */
    hipEvent& operator = (hipEvent&& rhs) {
      hipEventDeleter {} (_event);
      _event = rhs._event;
      rhs._event = nullptr;
      return *this;
    }
  
    /**
    @brief implicit conversion to the native hip event (hipEvent_t)

    Returns the underlying event of type @c hipEvent_t.
    */
    operator hipEvent_t () const {
      return _event;
    }
    
  private:

    hipEvent_t _event {nullptr};
};


}  // end of namespace tf -----------------------------------------------------



