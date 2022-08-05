// This is considered as "handcrafted version" using levelization
// to avoid all tasking overhead. 
// The performance is typically the best.
#include "graph.hpp"
#include <queue>

struct hipStream {

  std::vector<std::vector<hipStream_t>> streams;

  hipStream(unsigned N) : streams(tf::hip_get_num_devices()) {
    for(size_t i=0; i<streams.size(); ++i) {
      streams[i].resize(N);
      tf::hipScopedDevice ctx(i);
      for(unsigned j=0; j<N; ++j) {
        TF_CHECK_hip(hipStreamCreate(&streams[i][j]), "failed to create a stream on ", i);
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


void omp(const Graph& g, unsigned num_cpus, unsigned num_gpus) {

  std::vector<size_t> indegree(g.num_nodes, 0);
  std::vector<std::vector<int>> levellist(g.num_nodes);
  std::vector<std::vector<int>> adjlist(g.num_nodes);
  
  hipStream streams(num_cpus + num_gpus);
  
  for(const auto& e : g.edges) {
    adjlist[e.u].push_back(e.v);
    indegree[e.v]++;
  }

  std::queue<std::pair<int, int>> queue;

  for(size_t i=0; i<indegree.size(); ++i) {
    if(indegree[i] == 0) {
      queue.push({i, 0});
    }
  }

  while(!queue.empty()) {
    auto u = queue.front();
    queue.pop();

    levellist[u.second].push_back(u.first);
    
    for(auto v : adjlist[u.first]) {
      indegree[v]--;
      if(indegree[v] == 0) {
        queue.push({v, u.second + 1});
      }
    }
  }

  //std::cout << "Levellist:\n";
  //for(size_t l=0; l<levellist.size(); ++l) {
  //  std::cout << l << ':';
  //  for(const auto u : levellist[l]) {
  //    std::cout << ' ' << u;
  //  }
  //  std::cout << '\n';
  //}
  
  std::atomic<int> counter {0};
  
  int* cx = new int[N];
  int* cy = new int[N];
  int* cz = new int[N];
  int* gx = nullptr;
  int* gy = nullptr;
  int* gz = nullptr;
  TF_CHECK_hip(hipMallocManaged(&gx, N*sizeof(int)), "failed at hipMallocManaged");
  TF_CHECK_hip(hipMallocManaged(&gy, N*sizeof(int)), "failed at hipMallocManaged");
  TF_CHECK_hip(hipMallocManaged(&gz, N*sizeof(int)), "failed at hipMallocManaged");
  
  for(size_t l=0; l<levellist.size(); ++l) {
    #pragma omp parallel for num_threads(num_cpus + num_gpus)
    for(size_t i=0; i<levellist[l].size(); ++i) {
      int u = levellist[l][i];
      if(g.nodes[u].g== -1) {
        ++counter;
        for(int i=0; i<N; ++i) {
          cz[i] = cx[i] + cy[i];
        }
      }
      else {
        ++counter;
        int tgt_device = g.nodes[u].g;
        int src_device = -1;
        TF_CHECK_hip(hipGetDevice(&src_device), "get device failed");
        TF_CHECK_hip(hipSetDevice(tgt_device), "set device failed");

        hipGraph_t hip_graph;
        TF_CHECK_hip(hipGraphCreate(&hip_graph, 0), "hipGraphCreate failed");

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
        TF_CHECK_hip(hipGraphAddMemsetNode(&sgx, hip_graph, 0, 0, &msetp), "sgx failed");
        
        // sgy
        hipGraphNode_t sgy;
        msetp.dst = gy;
        TF_CHECK_hip(hipGraphAddMemsetNode(&sgy, hip_graph, 0, 0, &msetp), "sgy failed");
        
        // sgz
        hipGraphNode_t sgz;
        msetp.dst = gz;
        TF_CHECK_hip(hipGraphAddMemsetNode(&sgz, hip_graph, 0, 0, &msetp), "sgz failed");
      
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
        TF_CHECK_hip(hipGraphAddMemcpyNode(&h2d_gx, hip_graph, 0, 0, &h2dp), "h2d_gx failed");

        // h2d_gy
        hipGraphNode_t h2d_gy;
        h2dp.srcPtr = ::make_hipPitchedPtr(cy, N*sizeof(int), N, 1);
        h2dp.dstPtr = ::make_hipPitchedPtr(gy, N*sizeof(int), N, 1);
        TF_CHECK_hip(hipGraphAddMemcpyNode(&h2d_gy, hip_graph, 0, 0, &h2dp), "h2d_gy failed");

        // h2d_gz
        hipGraphNode_t h2d_gz;
        h2dp.srcPtr = ::make_hipPitchedPtr(cz, N*sizeof(int), N, 1);
        h2dp.dstPtr = ::make_hipPitchedPtr(gz, N*sizeof(int), N, 1);
        TF_CHECK_hip(hipGraphAddMemcpyNode(&h2d_gz, hip_graph, 0, 0, &h2dp), "h2d_gz failed");
      
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
        TF_CHECK_hip(hipGraphAddKernelNode(&kernel, hip_graph, 0, 0, &kp), "kernel failed");
        
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
        TF_CHECK_hip(hipGraphAddMemcpyNode(&d2h_gx, hip_graph, 0, 0, &d2hp), "d2h_gx failed");
        
        // d2h_gy
        hipGraphNode_t d2h_gy;
        d2hp.srcPtr = ::make_hipPitchedPtr(gy, N*sizeof(int), N, 1);
        d2hp.dstPtr = ::make_hipPitchedPtr(cy, N*sizeof(int), N, 1);
        TF_CHECK_hip(hipGraphAddMemcpyNode(&d2h_gy, hip_graph, 0, 0, &d2hp), "d2h_gy failed");
        
        // d2h_gz
        hipGraphNode_t d2h_gz;
        d2hp.srcPtr = ::make_hipPitchedPtr(gz, N*sizeof(int), N, 1);
        d2hp.dstPtr = ::make_hipPitchedPtr(cz, N*sizeof(int), N, 1);
        TF_CHECK_hip(hipGraphAddMemcpyNode(&d2h_gz, hip_graph, 0, 0, &d2hp), "d2h_gz failed");
      
        // add dependency
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &sgx, &h2d_gx, 1), "sgx->h2d_gx");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &sgy, &h2d_gy, 1), "sgy->h2d_gy");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &sgz, &h2d_gz, 1), "sgz->h2d_gz");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &h2d_gx, &kernel, 1), "h2d_gz->kernel");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &h2d_gy, &kernel, 1), "h2d_gz->kernel");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &h2d_gz, &kernel, 1), "h2d_gz->kernel");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &kernel, &d2h_gx, 1), "kernel->d2h_gx");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &kernel, &d2h_gy, 1), "kernel->d2h_gy");
        TF_CHECK_hip(hipGraphAddDependencies(hip_graph, &kernel, &d2h_gz, 1), "kernel->d2h_gz");
        
        // launch the graph
        hipStream_t pts = streams.per_thread_stream(tgt_device);

        hipGraphExec_t exe;
        TF_CHECK_hip(hipGraphInstantiate(&exe, hip_graph, 0, 0, 0), "inst failed");
        TF_CHECK_hip(hipGraphLaunch(exe, pts), "failed to launch hipGraph");
        TF_CHECK_hip(hipStreamSynchronize(pts), "failed to sync hipStream");
        TF_CHECK_hip(hipGraphExecDestroy(exe), "destroy exe failed");
        TF_CHECK_hip(hipGraphDestroy(hip_graph), "hipGraphDestroy failed");
        TF_CHECK_hip(hipSetDevice(src_device), "set device failed");
      }
    }

  }
  
  delete [] cx;
  delete [] cy;
  delete [] cz;
  TF_CHECK_hip(hipFree(gx), "failed at hipFree");
  TF_CHECK_hip(hipFree(gy), "failed at hipFree");
  TF_CHECK_hip(hipFree(gz), "failed at hipFree");

  if(counter != g.num_nodes) {
    throw std::runtime_error("wrong result");
  }
}


std::chrono::microseconds measure_time_omp(
  const Graph& g, unsigned num_cpus, unsigned num_gpus
) {
  auto beg = std::chrono::high_resolution_clock::now();
  omp(g, num_cpus, num_gpus);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
