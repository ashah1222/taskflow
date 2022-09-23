#pragma once

#include "hip_graph.hpp"

/**
@file hip_task.hpp
@brief hipTask include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// hipTask Types
// ----------------------------------------------------------------------------

/**
@enum hipTaskType

@brief enumeration of all %hipTask types
*/
enum class hipTaskType : int {
  /** @brief empty task type */
  EMPTY = 0,
  /** @brief host task type */
  HOST,
  /** @brief memory set task type */
  MEMSET,
  /** @brief memory copy task type */
  MEMCPY,
  /** @brief memory copy task type */
  KERNEL,
  /** @brief subflow (child graph) task type */
  SUBFLOW,
  /** @brief capture task type */
  CAPTURE,
  /** @brief undefined task type */
  UNDEFINED
};

/**
@brief convert a hip_task type to a human-readable string
*/
constexpr const char* to_string(hipTaskType type) {
  switch(type) {
    case hipTaskType::EMPTY:   return "empty";
    case hipTaskType::HOST:    return "host";
    case hipTaskType::MEMSET:  return "memset";
    case hipTaskType::MEMCPY:  return "memcpy";
    case hipTaskType::KERNEL:  return "kernel";
    case hipTaskType::SUBFLOW: return "subflow";
    case hipTaskType::CAPTURE: return "capture";
    default:                    return "undefined";
  }
}

// ----------------------------------------------------------------------------
// hipTask
// ----------------------------------------------------------------------------

/**
@class hipTask

@brief class to create a task handle over an internal node of a %hipFlow graph
*/
class hipTask {

  friend class hipFlow;
  friend class hipFlowCapturer;
  friend class hipFlowCapturerBase;

  friend std::ostream& operator << (std::ostream&, const hipTask&);

  public:

    /**
    @brief constructs an empty hipTask
    */
    hipTask() = default;

    /**
    @brief copy-constructs a hipTask
    */
    hipTask(const hipTask&) = default;

    /**
    @brief copy-assigns a hipTask
    */
    hipTask& operator = (const hipTask&) = default;

    /**
    @brief adds precedence links from this to other tasks

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    hipTask& precede(Ts&&... tasks);

    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    hipTask& succeed(Ts&&... tasks);

    /**
    @brief assigns a name to the task

    @param name a @std_string acceptable string

    @return @c *this
    */
    hipTask& name(const std::string& name);

    /**
    @brief queries the name of the task
    */
    const std::string& name() const;

    /**
    @brief queries the number of successors
    */
    size_t num_successors() const;

    /**
    @brief queries the number of dependents
    */
    size_t num_dependents() const;

    /**
    @brief queries if the task is associated with a hipNode
    */
    bool empty() const;

    /**
    @brief queries the task type
    */
    hipTaskType type() const;

    /**
    @brief dumps the task through an output stream

    @tparam T output stream type with insertion operator (<<) defined
    @param ostream an output stream target
    */
    template <typename T>
    void dump(T& ostream) const;

    /**
    @brief applies an visitor callable to each successor of the task
    */
    template <typename V>
    void for_each_successor(V&& visitor) const;

    /**
    @brief applies an visitor callable to each dependents of the task
    */
    template <typename V>
    void for_each_dependent(V&& visitor) const;

  private:

    hipTask(hipNode*);

    hipNode* _node {nullptr};
};

// Constructor
inline hipTask::hipTask(hipNode* node) : _node {node} {
}

// Function: precede
template <typename... Ts>
hipTask& hipTask::precede(Ts&&... tasks) {
  (_node->_precede(tasks._node), ...);
  return *this;
}

// Function: succeed
template <typename... Ts>
hipTask& hipTask::succeed(Ts&&... tasks) {
  (tasks._node->_precede(_node), ...);
  return *this;
}

// Function: empty
inline bool hipTask::empty() const {
  return _node == nullptr;
}

// Function: name
inline hipTask& hipTask::name(const std::string& name) {
  _node->_name = name;
  return *this;
}

// Function: name
inline const std::string& hipTask::name() const {
  return _node->_name;
}

// Function: num_successors
inline size_t hipTask::num_successors() const {
  return _node->_successors.size();
}

// Function: num_dependents
inline size_t hipTask::num_dependents() const {
  return _node->_dependents.size();
}

// Function: type
inline hipTaskType hipTask::type() const {
  switch(_node->_handle.index()) {
    case hipNode::EMPTY:   return hipTaskType::EMPTY;
    case hipNode::HOST:    return hipTaskType::HOST;
    case hipNode::MEMSET:  return hipTaskType::MEMSET;
    case hipNode::MEMCPY:  return hipTaskType::MEMCPY;
    case hipNode::KERNEL:  return hipTaskType::KERNEL;
    case hipNode::SUBFLOW: return hipTaskType::SUBFLOW;
    case hipNode::CAPTURE: return hipTaskType::CAPTURE;
    default:                return hipTaskType::UNDEFINED;
  }
}

// Procedure: dump
template <typename T>
void hipTask::dump(T& os) const {
  os << "hipTask ";
  if(_node->_name.empty()) os << _node;
  else os << _node->_name;
  os << " [type=" << to_string(type()) << ']';
}

// Function: for_each_successor
template <typename V>
void hipTask::for_each_successor(V&& visitor) const {
  for(size_t i=0; i<_node->_successors.size(); ++i) {
    visitor(hipTask(_node->_successors[i]));
  }
}

// Function: for_each_dependent
template <typename V>
void hipTask::for_each_dependent(V&& visitor) const {
  for(size_t i=0; i<_node->_dependents.size(); ++i) {
    visitor(hipTask(_node->_dependents[i]));
  }
}

// ----------------------------------------------------------------------------
// global ostream
// ----------------------------------------------------------------------------

/**
@brief overload of ostream inserter operator for hipTask
*/
inline std::ostream& operator << (std::ostream& os, const hipTask& ct) {
  ct.dump(os);
  return os;
}

}  // end of namespace tf -----------------------------------------------------



