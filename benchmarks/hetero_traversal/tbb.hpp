#include "graph.hpp"

#include <tbb/global_control.h>
#include <tbb/flow_graph.h>

struct hipStream {

  std::vector<std::vector<hipStream_t>> streams;

  hipStream(unsigned N) : streams(tf::hip_get_num_devices()) {
    for(size_t i=0; i<streams.size(); ++i) {
      streams[i].resize(N);
      tf::hipScopedDevice ctx(i);
      for(unsigned j=0; j<N; ++j) {
        TF_CHECK_HIP(hipStreamCreate(&streams[i][j]), "failed to create a stream on ", i);
      }
    }
  }

  ~hipStream() {
    for(size_t i=0; i<streams.size(); ++i) {
      tf::hipScopedDevice ctx(i);
      for(unsigned j=0; j<streams[i].size(); ++j) {
        hipStreamDestroy(streams[i][j]);
      }
    }
  }
  
  hipStream_t per_thread_stream(int device) {
    auto id = std::hash<std::thread::id>()(std::this_thread::get_id()) % streams[device].size();
    return streams[device][id];
  }

};

  
void TBB(const Graph& g, unsigned num_cpus, unsigned num_gpus) {

  using namespace tbb;
  using namespace tbb::flow;
    
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_cpus + num_gpus
  );
  tbb::flow::graph G;

  hipStream streams(num_cpus + num_gpus);
  
  std::atomic<int> counter {0};
  
  int* cx = new int[N];
  int* cy = new int[N];
  int* cz = new int[N];
  int* gx = nullptr;
  int* gy = nullptr;
  int* gz = nullptr;
  TF_CHECK_HIP(hipMallocManaged(&gx, N*sizeof(int)), "failed at hipMallocManaged");
  TF_CHECK_HIP(hipMallocManaged(&gy, N*sizeof(int)), "failed at hipMallocManaged");
  TF_CHECK_HIP(hipMallocManaged(&gz, N*sizeof(int)), "failed at hipMallocManaged");

  std::vector<std::unique_ptr<continue_node<continue_msg>>> tasks(g.num_nodes);
  std::vector<size_t> indegree(g.num_nodes, 0);
  auto source = std::make_unique<continue_node<continue_msg>>(
    G, [](const continue_msg&){}
  );
  
  // create a task for each node
  for(const auto& v : g.nodes) {
    // cpu task
    if(v.g == -1) {
      tasks[v.v] = std::make_unique<continue_node<continue_msg>>(G, 
        [&](const continue_msg&){ 
          for(int i=0; i<N; ++i) {
            cz[i] = cx[i] + cy[i];
          }
          ++counter; 
        }
      );
    }
    else {
      tasks[v.v] = std::make_unique<continue_node<continue_msg>>(G,
        [&, tgt_device=v.g](const continue_msg&){
          ++counter;
          int src_device = -1;
          TF_CHECK_HIP(hipGetDevice(&src_device), "get device failed");
          TF_CHECK_HIP(hipSetDevice(tgt_device), "set device failed");

          hipGraph_t hip_graph;
          TF_CHECK_HIP(hipGraphCreate(&hip_graph, 0), "hipGraphCreate failed");

          // memset parameter
          hipMemsetParams msetp;
          msetp.value = 0;
          msetp.pitch = 0;
          msetp.elementSize = sizeof(int);  // either 1, 2, or 4
          msetp.width = N;
          msetp.height = 1;

          // sgx
          hipGraphNode_t sgx;
          msetp.dst = gx;
          TF_CHECK_HIP(hipGraphAddMemsetNode(&sgx, hip_graph, 0, 0, &msetp), "sgx failed");
          
          // sgy
          hipGraphNode_t sgy;
          msetp.dst = gy;
          TF_CHECK_HIP(hipGraphAddMemsetNode(&sgy, hip_graph, 0, 0, &msetp), "sgy failed");
          
          // sgz
          hipGraphNode_t sgz;
          msetp.dst = gz;
          TF_CHECK_HIP(hipGraphAddMemsetNode(&sgz, hip_graph, 0, 0, &msetp), "sgz failed");
      
          // copy parameter
          hipMemcpy3DParms h2dp;
          h2dp.srcArray = nullptr;
          h2dp.srcPos = ::make_hipPos(0, 0, 0);
          h2dp.dstArray = nullptr;
          h2dp.dstPos = ::make_hipPos(0, 0, 0);
          h2dp.extent = ::make_hipExtent(N*sizeof(int), 1, 1);
          h2dp.kind = hipMemcpyDefault;

          // h2d_gx
          hipGraphNode_t h2d_gx;
          h2dp.srcPtr = ::make_hipPitchedPtr(cx, N*sizeof(int), N, 1);
          h2dp.dstPtr = ::make_hipPitchedPtr(gx, N*sizeof(int), N, 1);
          TF_CHECK_HIP(hipGraphAddMemcpyNode(&h2d_gx, hip_graph, 0, 0, &h2dp), "h2d_gx failed");

          // h2d_gy
          hipGraphNode_t h2d_gy;
          h2dp.srcPtr = ::make_hipPitchedPtr(cy, N*sizeof(int), N, 1);
          h2dp.dstPtr = ::make_hipPitchedPtr(gy, N*sizeof(int), N, 1);
          TF_CHECK_HIP(hipGraphAddMemcpyNode(&h2d_gy, hip_graph, 0, 0, &h2dp), "h2d_gy failed");

          // h2d_gz
          hipGraphNode_t h2d_gz;
          h2dp.srcPtr = ::make_hipPitchedPtr(cz, N*sizeof(int), N, 1);
          h2dp.dstPtr = ::make_hipPitchedPtr(gz, N*sizeof(int), N, 1);
          TF_CHECK_HIP(hipGraphAddMemcpyNode(&h2d_gz, hip_graph, 0, 0, &h2dp), "h2d_gz failed");
      
          // kernel
          hipKernelNodeParams kp;
          void* arguments[4] = { (void*)(&gx), (void*)(&gy), (void*)(&gz), (void*)(&N) };
          kp.func = (void*)add<int>;
          kp.gridDim = (N+255)/256;
          kp.blockDim = 256;
          kp.sharedMemBytes = 0;
          kp.kernelParams = arguments;
          kp.extra = nullptr;
          
          hipGraphNode_t kernel;
          TF_CHECK_HIP(hipGraphAddKernelNode(&kernel, hip_graph, 0, 0, &kp), "kernel failed");
          
          // d2hp
          hipMemcpy3DParms d2hp;
          d2hp.srcArray = nullptr;
          d2hp.srcPos = ::make_hipPos(0, 0, 0);
          d2hp.dstArray = nullptr;
          d2hp.dstPos = ::make_hipPos(0, 0, 0);
          d2hp.extent = ::make_hipExtent(N*sizeof(int), 1, 1);
          d2hp.kind = hipMemcpyDefault;
          
          // d2h_gx
          hipGraphNode_t d2h_gx;
          d2hp.srcPtr = ::make_hipPitchedPtr(gx, N*sizeof(int), N, 1);
          d2hp.dstPtr = ::make_hipPitchedPtr(cx, N*sizeof(int), N, 1);
          TF_CHECK_HIP(hipGraphAddMemcpyNode(&d2h_gx, hip_graph, 0, 0, &d2hp), "d2h_gx failed");
          
          // d2h_gy
          hipGraphNode_t d2h_gy;
          d2hp.srcPtr = ::make_hipPitchedPtr(gy, N*sizeof(int), N, 1);
          d2hp.dstPtr = ::make_hipPitchedPtr(cy, N*sizeof(int), N, 1);
          TF_CHECK_HIP(hipGraphAddMemcpyNode(&d2h_gy, hip_graph, 0, 0, &d2hp), "d2h_gy failed");
          
          // d2h_gz
          hipGraphNode_t d2h_gz;
          d2hp.srcPtr = ::make_hipPitchedPtr(gz, N*sizeof(int), N, 1);
          d2hp.dstPtr = ::make_hipPitchedPtr(cz, N*sizeof(int), N, 1);
          TF_CHECK_HIP(hipGraphAddMemcpyNode(&d2h_gz, hip_graph, 0, 0, &d2hp), "d2h_gz failed");
      
          // add dependency
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &sgx, &h2d_gx, 1), "sgx->h2d_gx");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &sgy, &h2d_gy, 1), "sgy->h2d_gy");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &sgz, &h2d_gz, 1), "sgz->h2d_gz");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &h2d_gx, &kernel, 1), "h2d_gz->kernel");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &h2d_gy, &kernel, 1), "h2d_gz->kernel");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &h2d_gz, &kernel, 1), "h2d_gz->kernel");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &kernel, &d2h_gx, 1), "kernel->d2h_gx");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &kernel, &d2h_gy, 1), "kernel->d2h_gy");
          TF_CHECK_HIP(hipGraphAddDependencies(hip_graph, &kernel, &d2h_gz, 1), "kernel->d2h_gz");
          
          // launch the graph
          hipGraphExec_t exe;

          auto pts = streams.per_thread_stream(tgt_device);

          TF_CHECK_HIP(hipGraphInstantiate(&exe, hip_graph, 0, 0, 0), "inst failed");
          TF_CHECK_HIP(hipGraphLaunch(exe, pts), "failed to launch hipGraph");
          TF_CHECK_HIP(hipStreamSynchronize(pts), "failed to sync hipStream");
          TF_CHECK_HIP(hipGraphExecDestroy(exe), "destroy exe failed");
          TF_CHECK_HIP(hipGraphDestroy(hip_graph), "hipGraphDestroy failed");
          TF_CHECK_HIP(hipSetDevice(src_device), "set device failed");
        }
      );
    }
  }
  for(const auto& e : g.edges) {
    make_edge(*tasks[e.u], *tasks[e.v]);
    indegree[e.v]++;
  }
  for(size_t i=0; i<indegree.size(); ++i) {
    if(indegree[i] == 0) {
      make_edge(*source, *tasks[i]);
    }
  }
  source->try_put(continue_msg());
  G.wait_for_all();

  
  delete [] cx;
  delete [] cy;
  delete [] cz;
  TF_CHECK_HIP(hipFree(gx), "failed at hipFree");
  TF_CHECK_HIP(hipFree(gy), "failed at hipFree");
  TF_CHECK_HIP(hipFree(gz), "failed at hipFree");
  
  if(counter != g.num_nodes) {
    throw std::runtime_error("wrong result");
  }

}

std::chrono::microseconds measure_time_tbb(
  const Graph& g, unsigned num_cpus, unsigned num_gpus
) {
  auto beg = std::chrono::high_resolution_clock::now();
  TBB(g, num_cpus, num_gpus);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

