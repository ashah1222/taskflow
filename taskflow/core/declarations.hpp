#pragma once

namespace tf {

// ----------------------------------------------------------------------------
// taskflow
// ----------------------------------------------------------------------------
class AsyncTopology;
class Node;
class Graph;
class FlowBuilder;
class Semaphore;
class Subflow;
class Runtime;
class Task;
class TaskView;
class Taskflow;
class Topology;
class TopologyBase;
class Executor;
class Worker;
class WorkerView;
class ObserverInterface;
class ChromeTracingObserver;
class TFProfObserver;
class TFProfManager;

template <typename T>
class Future;

template <typename...Fs>
class Pipeline;

// ----------------------------------------------------------------------------
// hipFlow
// ----------------------------------------------------------------------------
class hipNode;
class hipGraph;
class hipTask;
class hipFlow;
class hipFlowCapturer;
class hipFlowCapturerBase;
class hipCapturingBase;
class hipLinearCapturing;
class hipSequentialCapturing;
class hipRoundRobinCapturing;

// ----------------------------------------------------------------------------
// syclFlow
// ----------------------------------------------------------------------------
class syclNode;
class syclGraph;
class syclTask;
class syclFlow;


}  // end of namespace tf -----------------------------------------------------




