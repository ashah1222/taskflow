#pragma once

#include "hip_memory.hpp"
#include "hip_stream.hpp"
#include "hip_meta.hpp"

#include "../utility/traits.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// hipGraph_t routines
// ----------------------------------------------------------------------------

/**
@brief gets the memcpy node parameter of a copy task
*/
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
>
hipMemcpy3DParms hip_get_copy_parms(T* tgt, const T* src, size_t num) {

  using U = std::decay_t<T>;

  hipMemcpy3DParms p;

  p.srcArray = nullptr;
  p.srcPos = ::make_hipPos(0, 0, 0);
  p.srcPtr = ::make_hipPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_hipPos(0, 0, 0);
  p.dstPtr = ::make_hipPitchedPtr(tgt, num*sizeof(U), num, 1);
  p.extent = ::make_hipExtent(num*sizeof(U), 1, 1);
  p.kind = hipMemcpyDefault;

  return p;
}

/**
@brief gets the memcpy node parameter of a memcpy task (untyped)
*/
inline hipMemcpy3DParms hip_get_memcpy_parms(
  void* tgt, const void* src, size_t bytes
)  {

  // Parameters in hipPitchedPtr
  // d   - Pointer to allocated memory
  // p   - Pitch of allocated memory in bytes
  // xsz - Logical width of allocation in elements
  // ysz - Logical height of allocation in elements
  hipMemcpy3DParms p;
  p.srcArray = nullptr;
  p.srcPos = ::make_hipPos(0, 0, 0);
  p.srcPtr = ::make_hipPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_hipPos(0, 0, 0);
  p.dstPtr = ::make_hipPitchedPtr(tgt, bytes, bytes, 1);
  p.extent = ::make_hipExtent(bytes, 1, 1);
  p.kind = hipMemcpyDefault;

  return p;
}

/**
@brief gets the memset node parameter of a memcpy task (untyped)
*/
inline hipMemsetParams hip_get_memset_parms(void* dst, int ch, size_t count) {

  hipMemsetParams p;
  p.dst = dst;
  p.value = ch;
  p.pitch = 0;
  //p.elementSize = (count & 1) == 0 ? ((count & 3) == 0 ? 4 : 2) : 1;
  //p.width = (count & 1) == 0 ? ((count & 3) == 0 ? count >> 2 : count >> 1) : count;
  p.elementSize = 1;  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

/**
@brief gets the memset node parameter of a fill task (typed)
*/
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
>
hipMemsetParams hip_get_fill_parms(T* dst, T value, size_t count) {

  hipMemsetParams p;
  p.dst = dst;

  // perform bit-wise copy
  p.value = 0;  // crucial
  static_assert(sizeof(T) <= sizeof(p.value), "internal error");
  std::memcpy(&p.value, &value, sizeof(T));

  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

/**
@brief gets the memset node parameter of a zero task (typed)
*/
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
>
hipMemsetParams hip_get_zero_parms(T* dst, size_t count) {

  hipMemsetParams p;
  p.dst = dst;
  p.value = 0;
  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

/**
@brief queries the number of root nodes in a native hip graph
*/
inline size_t hip_get_graph_num_root_nodes(hipGraph_t graph) {
  size_t num_nodes;
  TF_CHECK_HIP(
    hipGraphGetRootNodes(graph, nullptr, &num_nodes),
    "failed to get native graph root nodes"
  );
  return num_nodes;
}

/**
@brief queries the number of nodes in a native hip graph
*/
inline size_t hip_get_graph_num_nodes(hipGraph_t graph) {
  size_t num_nodes;
  TF_CHECK_HIP(
    hipGraphGetNodes(graph, nullptr, &num_nodes),
    "failed to get native graph nodes"
  );
  return num_nodes;
}

/**
@brief queries the number of edges in a native hip graph
*/
inline size_t hip_get_graph_num_edges(hipGraph_t graph) {
  size_t num_edges;
  TF_CHECK_HIP(
    hipGraphGetEdges(graph, nullptr, nullptr, &num_edges),
    "failed to get native graph edges"
  );
  return num_edges;
}

/**
@brief acquires the nodes in a native hip graph
*/
inline std::vector<hipGraphNode_t> hip_get_graph_nodes(hipGraph_t graph) {
  size_t num_nodes = hip_get_graph_num_nodes(graph);
  std::vector<hipGraphNode_t> nodes(num_nodes);
  TF_CHECK_HIP(
    hipGraphGetNodes(graph, nodes.data(), &num_nodes),
    "failed to get native graph nodes"
  );
  return nodes;
}

/**
@brief acquires the root nodes in a native hip graph
*/
inline std::vector<hipGraphNode_t> hip_get_graph_root_nodes(hipGraph_t graph) {
  size_t num_nodes = hip_get_graph_num_root_nodes(graph);
  std::vector<hipGraphNode_t> nodes(num_nodes);
  TF_CHECK_HIP(
    hipGraphGetRootNodes(graph, nodes.data(), &num_nodes),
    "failed to get native graph nodes"
  );
  return nodes;
}

/**
@brief acquires the edges in a native hip graph
*/
inline std::vector<std::pair<hipGraphNode_t, hipGraphNode_t>>
hip_get_graph_edges(hipGraph_t graph) {
  size_t num_edges = hip_get_graph_num_edges(graph);
  std::vector<hipGraphNode_t> froms(num_edges), tos(num_edges);
  TF_CHECK_HIP(
    hipGraphGetEdges(graph, froms.data(), tos.data(), &num_edges),
    "failed to get native graph edges"
  );
  std::vector<std::pair<hipGraphNode_t, hipGraphNode_t>> edges(num_edges);
  for(size_t i=0; i<num_edges; i++) {
    edges[i] = std::make_pair(froms[i], tos[i]);
  }
  return edges;
}

/**
@brief queries the type of a native hip graph node

valid type values are:
  + hipGraphNodeTypeKernel      = 0x00
  + hipGraphNodeTypeMemcpy      = 0x01
  + hipGraphNodeTypeMemset      = 0x02
  + hipGraphNodeTypeHost        = 0x03
  + hipGraphNodeTypeGraph       = 0x04
  + hipGraphNodeTypeEmpty       = 0x05
  + hipGraphNodeTypeWaitEvent   = 0x06
  + hipGraphNodeTypeEventRecord = 0x07
*/
inline hipGraphNodeType hip_get_graph_node_type(hipGraphNode_t node) {
  hipGraphNodeType type;
  TF_CHECK_HIP(
    hipGraphNodeGetType(node, &type), "failed to get native graph node type"
  );
  return type;
}

/**
@brief convert the type of a native hip graph node to a readable string
*/
inline const char* hip_graph_node_type_to_string(hipGraphNodeType type) {
  switch(type) {
    case hipGraphNodeTypeKernel      : return "kernel";
    case hipGraphNodeTypeMemcpy      : return "memcpy";
    case hipGraphNodeTypeMemset      : return "memset";
    case hipGraphNodeTypeHost        : return "host";
    case hipGraphNodeTypeGraph       : return "graph";
    case hipGraphNodeTypeEmpty       : return "empty";
    case hipGraphNodeTypeWaitEvent   : return "event_wait";
    case hipGraphNodeTypeEventRecord : return "event_record";
    default                           : return "undefined";
  }
}

/**
@brief dumps a native hip graph and all associated child graphs to a DOT format

@tparam T output stream target
@param os target output stream
@param graph native hip graph
*/
template <typename T>
void hip_dump_graph(T& os, hipGraph_t graph) {

  os << "digraph hipGraph {\n";

  std::stack<std::tuple<hipGraph_t, hipGraphNode_t, int>> stack;
  stack.push(std::make_tuple(graph, nullptr, 1));

  int pl = 0;

  while(stack.empty() == false) {

    auto [graph, parent, l] = stack.top();
    stack.pop();

    for(int i=0; i<pl-l+1; i++) {
      os << "}\n";
    }

    os << "subgraph cluster_p" << graph << " {\n"
       << "label=\"hipGraph-L" << l << "\";\n"
       << "color=\"purple\";\n";

    auto nodes = hip_get_graph_nodes(graph);
    auto edges = hip_get_graph_edges(graph);

    for(auto& [from, to] : edges) {
      os << 'p' << from << " -> " << 'p' << to << ";\n";
    }

    for(auto& node : nodes) {
      auto type = hip_get_graph_node_type(node);
      if(type == hipGraphNodeTypeGraph) {

        hipGraph_t graph;
        TF_CHECK_HIP(hipGraphChildGraphNodeGetGraph(node, &graph), "");
        stack.push(std::make_tuple(graph, node, l+1));

        os << 'p' << node << "["
           << "shape=folder, style=filled, fontcolor=white, fillcolor=purple, "
           << "label=\"hipGraph-L" << l+1
           << "\"];\n";
      }
      else {
        os << 'p' << node << "[label=\""
           << hip_graph_node_type_to_string(type)
           << "\"];\n";
      }
    }

    // precede to parent
    if(parent != nullptr) {
      std::unordered_set<hipGraphNode_t> successors;
      for(const auto& p : edges) {
        successors.insert(p.first);
      }
      for(auto node : nodes) {
        if(successors.find(node) == successors.end()) {
          os << 'p' << node << " -> " << 'p' << parent << ";\n";
        }
      }
    }

    // set the previous level
    pl = l;
  }

  for(int i=0; i<=pl; i++) {
    os << "}\n";
  }
}

// ----------------------------------------------------------------------------
// hipGraphNative
// ----------------------------------------------------------------------------

/**
@class hipGraphNative

@brief class to create an RAII-styled wrapper over a hip executable graph

A hipGraphNative object is an RAII-styled wrapper over 
a native hip executable graph (@c hipGraphNative_t).
A hipGraphNative object is move-only.
*/
class hipGraphNative {

  struct hipGraphNativeCreator {
    hipGraph_t operator () () const { 
      hipGraph_t g;
      TF_CHECK_HIP(hipGraphCreate(&g, 0), "failed to create a hip native graph");
      return g; 
    }
  };
  
  struct hipGraphNativeDeleter {
    void operator () (hipGraph_t g) const {
      if(g) {
        hipGraphDestroy(g);
      }
    }
  };

  public:

    /**
    @brief constructs an RAII-styled object from the given hip exec

    Constructs a hipGraphNative object which owns @c exec.
    */
    explicit hipGraphNative(hipGraph_t native) : _native(native) {
    }
    
    /**
    @brief constructs an RAII-styled object for a new hip exec

    Equivalently calling @c hipGraphNativeCreate to create a exec.
    */
    hipGraphNative() : _native{ hipGraphNativeCreator{}() } {
    }
    
    /**
    @brief disabled copy constructor
    */
    hipGraphNative(const hipGraphNative&) = delete;
    
    /**
    @brief move constructor
    */
    hipGraphNative(hipGraphNative&& rhs) : _native{rhs._native} {
      rhs._native = nullptr;
    }

    /**
    @brief destructs the hip exec
    */
    ~hipGraphNative() {
      hipGraphNativeDeleter {} (_native);
    }
    
    /**
    @brief disabled copy assignment
    */
    hipGraphNative& operator = (const hipGraphNative&) = delete;

    /**
    @brief move assignment
    */
    hipGraphNative& operator = (hipGraphNative&& rhs) {
      hipGraphNativeDeleter {} (_native);
      _native = rhs._native;
      rhs._native = nullptr;
      return *this;
    }
    
    /**
    @brief implicit conversion to the native hip exec (hipGraphNative_t)

    Returns the underlying exec of type @c hipGraphNative_t.
    */
    operator hipGraph_t () const {
      return _native;
    }
    
  private:

    hipGraph_t _native {nullptr};
};

// ----------------------------------------------------------------------------
// hipGraphExec
// ----------------------------------------------------------------------------

/**
@class hipGraphExec

@brief class to create an RAII-styled wrapper over a hip executable graph

A hipGraphExec object is an RAII-styled wrapper over 
a native hip executable graph (@c hipGraphExec_t).
A hipGraphExec object is move-only.
*/
class hipGraphExec {

  struct hipGraphExecCreator {
    hipGraphExec_t operator () () const { return nullptr; }
  };
  
  struct hipGraphExecDeleter {
    void operator () (hipGraphExec_t executable) const {
      if(executable) {
        hipGraphExecDestroy(executable);
      }
    }
  };

  public:

    /**
    @brief constructs an RAII-styled object from the given hip exec

    Constructs a hipGraphExec object which owns @c exec.
    */
    explicit hipGraphExec(hipGraphExec_t exec) : _exec(exec) {
    }
    
    /**
    @brief constructs an RAII-styled object for a new hip exec

    Equivalently calling @c hipGraphExecCreate to create a exec.
    */
    hipGraphExec() : _exec{ hipGraphExecCreator{}() } {
    }
    
    /**
    @brief disabled copy constructor
    */
    hipGraphExec(const hipGraphExec&) = delete;
    
    /**
    @brief move constructor
    */
    hipGraphExec(hipGraphExec&& rhs) : _exec{rhs._exec} {
      rhs._exec = nullptr;
    }

    /**
    @brief destructs the hip exec
    */
    ~hipGraphExec() {
      hipGraphExecDeleter {} (_exec);
    }
    
    /**
    @brief disabled copy assignment
    */
    hipGraphExec& operator = (const hipGraphExec&) = delete;

    /**
    @brief move assignment
    */
    hipGraphExec& operator = (hipGraphExec&& rhs) {
      hipGraphExecDeleter {} (_exec);
      _exec = rhs._exec;
      rhs._exec = nullptr;
      return *this;
    }
    
    /**
    @brief replaces the managed executable graph with the given one

    Destructs the managed exec and resets it to the given exec.
    */
    void clear() {
      hipGraphExecDeleter {} (_exec);
      _exec = nullptr;
    }
    
    /**
    @brief instantiates the exexutable from the given hip graph
    */
    void instantiate(hipGraph_t graph) {
      hipGraphExecDeleter {} (_exec);
      TF_CHECK_HIP(
        hipGraphInstantiate(&_exec, graph, nullptr, nullptr, 0),
        "failed to create an executable graph"
      );
    }
    
    /**
    @brief updates the exexutable from the given hip graph
    */
    hipGraphExecUpdateResult update(hipGraph_t graph) {
      hipGraphNode_t error_node;
      hipGraphExecUpdateResult error_result;
      hipGraphExecUpdate(_exec, graph, &error_node, &error_result);
      return error_result;
    }
    
    /**
    @brief launchs the executable graph via the given stream
    */
    void launch(hipStream_t stream) {
      TF_CHECK_HIP(
        hipGraphLaunch(_exec, stream), "failed to launch a hip executable graph"
      );
    }
  
    /**
    @brief implicit conversion to the native hip exec (hipGraphExec_t)

    Returns the underlying exec of type @c hipGraphExec_t.
    */
    operator hipGraphExec_t () const {
      return _exec;
    }
    
  private:

    hipGraphExec_t _exec {nullptr};
};

// ----------------------------------------------------------------------------
// hipGraph class
// ----------------------------------------------------------------------------

// class: hipGraph
class hipGraph : public CustomGraphBase {

  friend class hipNode;
  friend class hipTask;
  friend class hipFlowCapturerBase;
  friend class hipFlowCapturer;
  friend class hipFlow;
  friend class hipCapturingBase;
  friend class hipSequentialCapturing;
  friend class hipLinearCapturing;
  friend class hipRoundRobinCapturing;
  friend class Taskflow;
  friend class Executor;

  constexpr static int OFFLOADED = 0x01;
  constexpr static int CHANGED   = 0x02;
  constexpr static int UPDATED   = 0x04;

  public:

    hipGraph() = default;
    ~hipGraph();

    hipGraph(const hipGraph&) = delete;
    hipGraph(hipGraph&&);

    hipGraph& operator = (const hipGraph&) = delete;
    hipGraph& operator = (hipGraph&&);

    template <typename... ArgsT>
    hipNode* emplace_back(ArgsT&&...);

    bool empty() const;

    void clear();
    void dump(std::ostream&, const void*, const std::string&) const override final;

  private:

    int _state{CHANGED};

    hipGraph_t _native_handle {nullptr};

    std::vector<std::unique_ptr<hipNode>> _nodes;
    //std::vector<hipNode*> _nodes;
};

// ----------------------------------------------------------------------------
// hipNode class
// ----------------------------------------------------------------------------

/**
@private
@class: hipNode
*/
class hipNode {

  friend class hipGraph;
  friend class hipTask;
  friend class hipFlow;
  friend class hipFlowCapturer;
  friend class hipFlowCapturerBase;
  friend class hipCapturingBase;
  friend class hipSequentialCapturing;
  friend class hipLinearCapturing;
  friend class hipRoundRobinCapturing;
  friend class Taskflow;
  friend class Executor;

  // Empty handle
  struct Empty {
  };

  // Host handle
  struct Host {

    template <typename C>
    Host(C&&);

    std::function<void()> func;

    static void callback(void*);
  };

  // Memset handle
  struct Memset {
  };

  // Memcpy handle
  struct Memcpy {
  };

  // Kernel handle
  struct Kernel {

    template <typename F>
    Kernel(F&& f);

    void* func {nullptr};
  };

  // Subflow handle
  struct Subflow {
    hipGraph graph;
  };

  // Capture
  struct Capture {

    template <typename C>
    Capture(C&&);

    std::function<void(hipStream_t)> work;

    hipEvent_t event;
    size_t level;
    size_t lid;
    size_t idx;
  };

  using handle_t = std::variant<
    Empty,
    Host,
    Memset,
    Memcpy,
    Kernel,
    Subflow,
    Capture
  >;

  public:

  // variant index
  constexpr static auto EMPTY   = get_index_v<Empty, handle_t>;
  constexpr static auto HOST    = get_index_v<Host, handle_t>;
  constexpr static auto MEMSET  = get_index_v<Memset, handle_t>;
  constexpr static auto MEMCPY  = get_index_v<Memcpy, handle_t>;
  constexpr static auto KERNEL  = get_index_v<Kernel, handle_t>;
  constexpr static auto SUBFLOW = get_index_v<Subflow, handle_t>;
  constexpr static auto CAPTURE = get_index_v<Capture, handle_t>;

    hipNode() = delete;

    template <typename... ArgsT>
    hipNode(hipGraph&, ArgsT&&...);

  private:

    hipGraph& _graph;

    std::string _name;

    handle_t _handle;

    hipGraphNode_t _native_handle {nullptr};

    SmallVector<hipNode*> _successors;
    SmallVector<hipNode*> _dependents;

    void _precede(hipNode*);
};

// ----------------------------------------------------------------------------
// hipNode definitions
// ----------------------------------------------------------------------------

// Host handle constructor
template <typename C>
hipNode::Host::Host(C&& c) : func {std::forward<C>(c)} {
}

// Host callback
inline void hipNode::Host::callback(void* data) {
  static_cast<Host*>(data)->func();
};

// Kernel handle constructor
template <typename F>
hipNode::Kernel::Kernel(F&& f) :
  func {std::forward<F>(f)} {
}

// Capture handle constructor
template <typename C>
hipNode::Capture::Capture(C&& work) :
  work {std::forward<C>(work)} {
}

// Constructor
template <typename... ArgsT>
hipNode::hipNode(hipGraph& graph, ArgsT&&... args) :
  _graph {graph},
  _handle {std::forward<ArgsT>(args)...} {
}

// Procedure: _precede
inline void hipNode::_precede(hipNode* v) {

  _graph._state |= hipGraph::CHANGED;

  _successors.push_back(v);
  v->_dependents.push_back(this);

  // capture node doesn't have the native graph yet
  if(_handle.index() != hipNode::CAPTURE) {
    TF_CHECK_HIP(
      hipGraphAddDependencies(
        _graph._native_handle, &_native_handle, &v->_native_handle, 1
      ),
      "failed to add a preceding link ", this, "->", v
    );
  }
}

//// Procedure: _set_state
//inline void hipNode::_set_state(int flag) {
//  _state |= flag;
//}
//
//// Procedure: _unset_state
//inline void hipNode::_unset_state(int flag) {
//  _state &= ~flag;
//}
//
//// Procedure: _clear_state
//inline void hipNode::_clear_state() {
//  _state = 0;
//}
//
//// Function: _has_state
//inline bool hipNode::_has_state(int flag) const {
//  return _state & flag;
//}

// ----------------------------------------------------------------------------
// hipGraph definitions
// ----------------------------------------------------------------------------

// Destructor
inline hipGraph::~hipGraph() {
  //clear();
  assert(_native_handle == nullptr);
}

// Move constructor
inline hipGraph::hipGraph(hipGraph&& g) :
  _native_handle {g._native_handle},
  _nodes         {std::move(g._nodes)} {

  g._native_handle = nullptr;

  assert(g._nodes.empty());
}

// Move assignment
inline hipGraph& hipGraph::operator = (hipGraph&& rhs) {

  //clear();

  // lhs
  _native_handle = rhs._native_handle;
  _nodes = std::move(rhs._nodes);

  assert(rhs._nodes.empty());

  // rhs
  rhs._native_handle = nullptr;

  return *this;
}

// Function: empty
inline bool hipGraph::empty() const {
  return _nodes.empty();
}

// Procedure: clear
inline void hipGraph::clear() {
  //for(auto n : _nodes) {
  //  delete n;
  //}
  _state = hipGraph::CHANGED;
  _nodes.clear();
}

// Function: emplace_back
template <typename... ArgsT>
hipNode* hipGraph::emplace_back(ArgsT&&... args) {

  _state |= hipGraph::CHANGED;

  auto node = std::make_unique<hipNode>(std::forward<ArgsT>(args)...);
  _nodes.emplace_back(std::move(node));
  return _nodes.back().get();

  // TODO: use object pool to save memory
  //auto node = new hipNode(std::forward<ArgsT>(args)...);
  //_nodes.push_back(node);
  //return node;
}

// Procedure: dump the graph to a DOT format
inline void hipGraph::dump(
  std::ostream& os, const void* root, const std::string& root_name
) const {

  // recursive dump with stack
  std::stack<std::tuple<const hipGraph*, const hipNode*, int>> stack;
  stack.push(std::make_tuple(this, nullptr, 1));

  int pl = 0;

  while(!stack.empty()) {

    auto [graph, parent, l] = stack.top();
    stack.pop();

    for(int i=0; i<pl-l+1; i++) {
      os << "}\n";
    }

    if(parent == nullptr) {
      if(root) {
        os << "subgraph cluster_p" << root << " {\nlabel=\"hipFlow: ";
        if(root_name.empty()) os << 'p' << root;
        else os << root_name;
        os << "\";\n" << "color=\"purple\"\n";
      }
      else {
        os << "digraph hipFlow {\n";
      }
    }
    else {
      os << "subgraph cluster_p" << parent << " {\nlabel=\"hipSubflow: ";
      if(parent->_name.empty()) os << 'p' << parent;
      else os << parent->_name;
      os << "\";\n" << "color=\"purple\"\n";
    }

    for(auto& node : graph->_nodes) {

      auto v = node.get();

      os << 'p' << v << "[label=\"";
      if(v->_name.empty()) {
        os << 'p' << v << "\"";
      }
      else {
        os << v->_name << "\"";
      }

      switch(v->_handle.index()) {
        case hipNode::KERNEL:
          os << " style=\"filled\""
             << " color=\"white\" fillcolor=\"black\""
             << " fontcolor=\"white\""
             << " shape=\"box3d\"";
        break;

        case hipNode::SUBFLOW:
          stack.push(std::make_tuple(
            &(std::get_if<hipNode::Subflow>(&v->_handle)->graph), v, l+1)
          );
          os << " style=\"filled\""
             << " color=\"black\" fillcolor=\"purple\""
             << " fontcolor=\"white\""
             << " shape=\"folder\"";
        break;

        default:
        break;
      }

      os << "];\n";

      for(const auto s : v->_successors) {
        os << 'p' << v << " -> " << 'p' << s << ";\n";
      }

      if(v->_successors.size() == 0) {
        if(parent == nullptr) {
          if(root) {
            os << 'p' << v << " -> p" << root << ";\n";
          }
        }
        else {
          os << 'p' << v << " -> p" << parent << ";\n";
        }
      }
    }

    // set the previous level
    pl = l;
  }

  for(int i=0; i<pl; i++) {
    os << "}\n";
  }

}


}  // end of namespace tf -----------------------------------------------------




