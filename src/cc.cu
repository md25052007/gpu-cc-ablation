// cc.cu — driver for the 4-way ablation (V0-V3) and the refined version V4.
//
//   V0  baseline        : s = 1,          one thread per vertex
//   V1  Sutton-style    : s = 2|E|/|V|,   one thread per vertex   (sliced neighbour lists)
//   V2  ECL-CC-style    : s = 1,          thread/warp/block by degree
//   V3  proposed        : s = 2|E|/|V|,   thread/warp/block by in-segment degree
//   V4  refined hybrid  : s = 2|E|/|V|,   light vertices: their edges hooked one thread per
//                                         edge from a flat light-edge list (contiguous chunks,
//                                         no idle threads); heavy vertices: sliced neighbour lists
//                                         with warp/block dispatch (as in V3)
//
// Every version: init -> for k in 0..s-1 { hook(segment k); Multi-Jump(all) } -> labels.
//
// Usage: cc <graph.egr> [options]   (run with --help)
#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "graph.h"
#include "kernels.cuh"

#define CK(x)                                                                   \
  do {                                                                          \
    cudaError_t e_ = (x);                                                       \
    if (e_ != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

enum Ver { V0, V1, V2, V3, V4 };
static const int NVER = 5;
static const char* VNAME[] = {"V0", "V1", "V2", "V3", "V4"};
static bool segmented(Ver v) { return v == V1 || v == V3 || v == V4; }
static bool bucketed(Ver v) { return v == V2 || v == V3 || v == V4; }

struct Config {
  std::string graph, csv, seg_csv, tag;
  std::vector<Ver> versions{V0, V1, V2, V3, V4};
  int segments = 0;  // 0 = auto (Sutton heuristic)
  int t1 = 4, t2 = 32, block = 256;
  int runs = 5, warmup = 1;
  bool verify = true, emulate = false, stats_only = false;
};

struct RunResult {
  double total_ms = 0, setup_ms = 0, hook_ms = 0, jump_ms = 0;
  int s = 1;
  long long nthr = 0, nwarp = 0, nblk = 0;
  long long light_edges = -1;  // V4 only: edges hooked one thread per edge
  size_t mem_bytes = 0;
  DevCounters ctr{};
  bool have_ctr = false;
  std::vector<int> labels;
  std::vector<DevCounters> per_seg;  // counter build only
};

// ------------------------------------------------------------ helpers
static int auto_segments(const ForwardCSR& f) {
  // Sutton et al.: s ~ average degree = 2|E| / |V|
  double s = 2.0 * (double)f.m / (double)f.n;
  int r = (int)std::lround(s);
  return r < 1 ? 1 : r;
}

static unsigned grid_for(long long threads, int block) {
  long long g = (threads + block - 1) / block;
  if (g > 2147483647LL) die("grid too large");
  return (unsigned)g;
}

struct InBucket {
  const int* off; int s, t1, t2, want;
  __host__ __device__ bool operator()(int v) const {
    return bucket_of(off[v + 1] - off[v], s, t1, t2) == want;
  }
};

// ------------------------------------------------------------ device graph
struct DeviceGraph {
  int n = 0; long long m = 0;
  int *off = nullptr, *adj = nullptr, *parent = nullptr, *wl = nullptr;
  int *lpos = nullptr, *lsrc = nullptr, *ldst = nullptr;  // V4 light-edge list
  void upload(const ForwardCSR& f, bool need_light) {
    n = f.n; m = f.m;
    CK(cudaMalloc(&off, sizeof(int) * (n + 1)));
    CK(cudaMalloc(&adj, sizeof(int) * (m > 0 ? m : 1)));
    CK(cudaMalloc(&parent, sizeof(int) * n));
    CK(cudaMalloc(&wl, sizeof(int) * n));
    CK(cudaMemcpy(off, f.off.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    if (m) CK(cudaMemcpy(adj, f.adj.data(), sizeof(int) * m, cudaMemcpyHostToDevice));
    if (need_light) {  // worst case: every edge is light
      CK(cudaMalloc(&lpos, sizeof(int) * (n + 1)));
      CK(cudaMalloc(&lsrc, sizeof(int) * (m > 0 ? m : 1)));
      CK(cudaMalloc(&ldst, sizeof(int) * (m > 0 ? m : 1)));
    }
  }
  void release() {
    cudaFree(off); cudaFree(adj); cudaFree(parent); cudaFree(wl);
    if (lpos) { cudaFree(lpos); cudaFree(lsrc); cudaFree(ldst); }
  }
};

// Bytes a version actually needs (reported metric, independent of what else is allocated).
static size_t mem_needed(Ver v, int n, long long m) {
  size_t b = sizeof(int) * ((size_t)n + 1 + (size_t)m + (size_t)n);  // off + adj + parent
  if (bucketed(v)) b += sizeof(int) * (size_t)n;                     // worklists
  if (v == V4) b += sizeof(int) * ((size_t)n + 1);                    // light offsets
  return b;                                                           // (+ 2 x light edges, added at run time)
}

// ------------------------------------------------------------ GPU run
static RunResult run_gpu(DeviceGraph& g, Ver ver, int s, const Config& cfg, bool want_seg) {
  RunResult r;
  r.s = s;
  r.mem_bytes = mem_needed(ver, g.n, g.m);
  const int B = cfg.block;
#ifdef CC_COUNTERS
  DevCounters zero{};
  CK(cudaMemcpyToSymbol(d_ctr, &zero, sizeof(zero)));
  DevCounters prev{};
#endif
  cudaEvent_t e_start, e_setup;
  CK(cudaEventCreate(&e_start)); CK(cudaEventCreate(&e_setup));
  std::vector<cudaEvent_t> e_hook(s), e_jump(s);
  for (int k = 0; k < s; k++) { CK(cudaEventCreate(&e_hook[k])); CK(cudaEventCreate(&e_jump[k])); }

  CK(cudaEventRecord(e_start));
  k_init<<<grid_for(g.n, B), B>>>(g.n, g.parent);
  long long c0 = 0, c1 = 0, c2 = 0;
  if (bucketed(ver)) {
    // Buckets use in-segment degree ceil(d/s); it differs by at most 1 between
    // segments, so the worklists are built ONCE per run (counted in setup time).
    auto first = thrust::make_counting_iterator<int>(0);
    auto last = first + g.n;
    int* p = g.wl;
    int* q = p;
    if (ver != V4) {  // V4 hooks light vertices through the edge list, not a vertex worklist
      q = thrust::copy_if(thrust::device, first, last, p, InBucket{g.off, s, cfg.t1, cfg.t2, 0});
      c0 = q - p; p = q;
    }
    q = thrust::copy_if(thrust::device, first, last, p, InBucket{g.off, s, cfg.t1, cfg.t2, 1});
    c1 = q - p; p = q;
    q = thrust::copy_if(thrust::device, first, last, p, InBucket{g.off, s, cfg.t1, cfg.t2, 2});
    c2 = q - p;
  }
  long long mL = 0;
  if (ver == V4) {
    // light-edge list: pos = exclusive scan of light degree, then copy the edges
    CK(cudaMemset(g.lpos + g.n, 0, sizeof(int)));
    auto first = thrust::make_counting_iterator<int>(0);
    thrust::transform(thrust::device, first, first + g.n, g.lpos, LightDeg{g.off, s, cfg.t1, cfg.t2});
    thrust::exclusive_scan(thrust::device, g.lpos, g.lpos + g.n + 1, g.lpos);
    k_build_light<<<grid_for((long long)g.n * LIGHT_GROUP, B), B>>>(g.n, g.off, g.adj, g.lpos, g.lsrc, g.ldst);
    int t;
    CK(cudaMemcpy(&t, g.lpos + g.n, sizeof(int), cudaMemcpyDeviceToHost));
    mL = t;
    c0 = g.n - c1 - c2;  // reported as the number of light vertices
    r.light_edges = mL;
    r.mem_bytes += 2 * sizeof(int) * (size_t)mL;
  }
  CK(cudaEventRecord(e_setup));
  r.nthr = c0; r.nwarp = c1; r.nblk = c2;
  const int *wl0 = g.wl, *wl1 = g.wl + (ver == V4 ? 0 : c0), *wl2 = wl1 + c1;

  for (int k = 0; k < s; k++) {
    switch (ver) {
      case V0: case V1:
        k_hook_vertex_all<<<grid_for(g.n, B), B>>>(g.n, g.off, g.adj, g.parent, k, s);
        break;
      case V2: case V3:
        if (c0) k_hook_thread<<<grid_for(c0, B), B>>>((int)c0, wl0, g.off, g.adj, g.parent, k, s);
        if (c1) k_hook_warp<<<grid_for(c1 * 32, B), B>>>((int)c1, wl1, g.off, g.adj, g.parent, k, s);
        if (c2) k_hook_block<<<(unsigned)c2, B>>>(wl2, g.off, g.adj, g.parent, k, s);
        break;
      case V4: {
        long long eb = mL * k / s, ee = mL * (k + 1) / s;
        if (ee > eb) k_hook_light<<<grid_for(ee - eb, B), B>>>(eb, ee, g.lsrc, g.ldst, g.parent);
        if (c1) k_hook_warp<<<grid_for(c1 * 32, B), B>>>((int)c1, wl1, g.off, g.adj, g.parent, k, s);
        if (c2) k_hook_block<<<(unsigned)c2, B>>>(wl2, g.off, g.adj, g.parent, k, s);
        break;
      }
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(e_hook[k]));
    k_multijump<<<grid_for(g.n, B), B>>>(g.n, g.parent);
    CK(cudaEventRecord(e_jump[k]));
#ifdef CC_COUNTERS
    if (want_seg) {  // per-segment deltas (synchronizes; counter build is never used for timing)
      DevCounters cur;
      CK(cudaMemcpyFromSymbol(&cur, d_ctr, sizeof(cur)));
      DevCounters d = cur;
      d.cas_attempts -= prev.cas_attempts; d.cas_retries -= prev.cas_retries;
      d.find_steps -= prev.find_steps; d.lane_work -= prev.lane_work;
      d.lane_slots -= prev.lane_slots; d.busy_lanes -= prev.busy_lanes;
      d.idle_threads -= prev.idle_threads;
      r.per_seg.push_back(d);
      prev = cur;
    }
#else
    (void)want_seg;
#endif
  }
  CK(cudaEventSynchronize(e_jump[s - 1]));
  CK(cudaGetLastError());

  float ms;
  CK(cudaEventElapsedTime(&ms, e_start, e_setup)); r.setup_ms = ms;
  cudaEvent_t prev_ev = e_setup;
  for (int k = 0; k < s; k++) {
    CK(cudaEventElapsedTime(&ms, prev_ev, e_hook[k])); r.hook_ms += ms;
    CK(cudaEventElapsedTime(&ms, e_hook[k], e_jump[k])); r.jump_ms += ms;
    prev_ev = e_jump[k];
  }
  CK(cudaEventElapsedTime(&ms, e_start, e_jump[s - 1])); r.total_ms = ms;

#ifdef CC_COUNTERS
  CK(cudaMemcpyFromSymbol(&r.ctr, d_ctr, sizeof(r.ctr)));
  r.have_ctr = true;
#endif
  r.labels.resize(g.n);
  CK(cudaMemcpy(r.labels.data(), g.parent, sizeof(int) * g.n, cudaMemcpyDeviceToHost));

  cudaEventDestroy(e_start); cudaEventDestroy(e_setup);
  for (int k = 0; k < s; k++) { cudaEventDestroy(e_hook[k]); cudaEventDestroy(e_jump[k]); }
  return r;
}

// ------------------------------------------------------------ CPU emulation
// Runs the same per-thread bodies sequentially (thread, every lane of a warp,
// every thread of a block). Checks that each version hooks every edge exactly
// once and produces the right partition. Says nothing about GPU races/timing.
static RunResult run_emulated(const ForwardCSR& f, Ver ver, int s, const Config& cfg) {
  RunResult r;
  r.s = s;
  r.mem_bytes = mem_needed(ver, f.n, f.m);
  const int* off = f.off.data();
  const int* adj = f.adj.data();
  std::vector<int> parent(f.n);
  auto t0 = std::chrono::steady_clock::now();
  for (int v = 0; v < f.n; v++) parent[v] = v;
  std::vector<int> wl[3];
  if (bucketed(ver))
    for (int v = 0; v < f.n; v++) wl[bucket_of(off[v + 1] - off[v], s, cfg.t1, cfg.t2)].push_back(v);
  r.nthr = wl[0].size(); r.nwarp = wl[1].size(); r.nblk = wl[2].size();
  std::vector<int> lsrc, ldst;  // V4 light-edge list, same layout as the GPU build
  if (ver == V4) {
    for (int v : wl[0]) for (int j = off[v]; j < off[v + 1]; j++) { lsrc.push_back(v); ldst.push_back(adj[j]); }
    r.light_edges = (long long)lsrc.size();
    r.mem_bytes += 2 * sizeof(int) * lsrc.size();
  }
  const long long mL = (long long)lsrc.size();
  LocalCtr c;
  long long hooked = 0;
  for (int k = 0; k < s; k++) {
    int b, e;
    switch (ver) {
      case V0: case V1:
        for (int v = 0; v < f.n; v++) { slice_of(off, v, k, s, b, e); hooked += hook_range(adj, parent.data(), v, b, e, 0, 1, c); }
        break;
      case V2: case V3:
        for (int v : wl[0]) { slice_of(off, v, k, s, b, e); hooked += hook_range(adj, parent.data(), v, b, e, 0, 1, c); }
        for (int v : wl[1]) { slice_of(off, v, k, s, b, e); for (int lane = 0; lane < 32; lane++) hooked += hook_range(adj, parent.data(), v, b, e, lane, 32, c); }
        for (int v : wl[2]) { slice_of(off, v, k, s, b, e); for (int t = 0; t < cfg.block; t++) hooked += hook_range(adj, parent.data(), v, b, e, t, cfg.block, c); }
        break;
      case V4: {
        long long eb = mL * k / s, ee = mL * (k + 1) / s;
        for (long long i = eb; i < ee; i++) { atomic_hook(parent.data(), lsrc[i], ldst[i], c); hooked++; }
        for (int v : wl[1]) { slice_of(off, v, k, s, b, e); for (int lane = 0; lane < 32; lane++) hooked += hook_range(adj, parent.data(), v, b, e, lane, 32, c); }
        for (int v : wl[2]) { slice_of(off, v, k, s, b, e); for (int t = 0; t < cfg.block; t++) hooked += hook_range(adj, parent.data(), v, b, e, t, cfg.block, c); }
        break;
      }
    }
    for (int v = 0; v < f.n; v++) multijump_vertex(parent.data(), v);
  }
  r.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (hooked != f.m) {
    fprintf(stderr, "EMULATION ERROR: %s hooked %lld edges, expected %lld\n", VNAME[ver], hooked, f.m);
    exit(2);
  }
  // after the last Multi-Jump every vertex must point directly at its root
  for (int v = 0; v < f.n; v++)
    if (parent[parent[v]] != parent[v] || parent[v] > v) {
      fprintf(stderr, "EMULATION ERROR: %s not flattened at v=%d\n", VNAME[ver], v);
      exit(2);
    }
  r.ctr.cas_attempts = c.att; r.ctr.cas_retries = c.ret; r.ctr.find_steps = c.steps;
  r.labels = parent;
  return r;
}

// ------------------------------------------------------------ output
static std::string base_name(const std::string& p) {
  size_t a = p.find_last_of("/\\");
  std::string b = a == std::string::npos ? p : p.substr(a + 1);
  size_t d = b.rfind('.');
  return d == std::string::npos ? b : b.substr(0, d);
}

static void csv_row(const Config& cfg, const std::string& gname, Ver v, int run,
                    const RunResult& r, int correct, long long comps) {
  if (cfg.csv.empty()) return;
  bool fresh = false;
  if (FILE* t = fopen(cfg.csv.c_str(), "r")) fclose(t); else fresh = true;
  FILE* f = fopen(cfg.csv.c_str(), "a");
  if (!f) die("cannot open csv");
  if (fresh)
    fprintf(f, "graph,tag,version,run,segments,t1,t2,block,total_ms,setup_ms,hook_ms,jump_ms,"
               "n_thread,n_warp,n_block,mem_bytes,correct,components,emulated,"
               "cas_attempts,cas_retries,find_steps,lane_work,lane_slots,busy_lanes,"
               "max_lane_work,idle_threads,simt_efficiency,imbalance,light_edges\n");
  fprintf(f, "%s,%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%lld,%lld,%lld,%zu,%d,%lld,%d,",
          gname.c_str(), cfg.tag.c_str(), VNAME[v], run, r.s, cfg.t1, cfg.t2, cfg.block,
          r.total_ms, r.setup_ms, r.hook_ms, r.jump_ms, r.nthr, r.nwarp, r.nblk,
          r.mem_bytes, correct, comps, cfg.emulate ? 1 : 0);
  if (r.have_ctr) {
    const DevCounters& c = r.ctr;
    double eff = c.lane_slots ? (double)c.lane_work / c.lane_slots : 0;
    double avg = c.busy_lanes ? (double)c.lane_work / c.busy_lanes : 0;
    double imb = avg > 0 ? c.max_lane_work / avg : 0;
    fprintf(f, "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.6f,%.4f,", c.cas_attempts, c.cas_retries,
            c.find_steps, c.lane_work, c.lane_slots, c.busy_lanes, c.max_lane_work,
            c.idle_threads, eff, imb);
  } else if (cfg.emulate) {
    fprintf(f, "%llu,%llu,%llu,,,,,,,,", r.ctr.cas_attempts, r.ctr.cas_retries, r.ctr.find_steps);
  } else {
    fprintf(f, ",,,,,,,,,,");
  }
  if (r.light_edges >= 0) fprintf(f, "%lld\n", r.light_edges); else fprintf(f, "\n");
  fclose(f);
}

static void seg_csv_rows(const Config& cfg, const std::string& gname, Ver v, const RunResult& r) {
  if (cfg.seg_csv.empty() || r.per_seg.empty()) return;
  bool fresh = false;
  if (FILE* t = fopen(cfg.seg_csv.c_str(), "r")) fclose(t); else fresh = true;
  FILE* f = fopen(cfg.seg_csv.c_str(), "a");
  if (!f) die("cannot open seg csv");
  if (fresh)
    fprintf(f, "graph,tag,version,segments,segment,cas_attempts,cas_retries,find_steps,"
               "lane_work,lane_slots,busy_lanes,idle_threads,simt_efficiency\n");
  for (size_t k = 0; k < r.per_seg.size(); k++) {
    const DevCounters& c = r.per_seg[k];
    double eff = c.lane_slots ? (double)c.lane_work / c.lane_slots : 0;
    fprintf(f, "%s,%s,%s,%d,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.6f\n", gname.c_str(),
            cfg.tag.c_str(), VNAME[v], r.s, k, c.cas_attempts, c.cas_retries, c.find_steps,
            c.lane_work, c.lane_slots, c.busy_lanes, c.idle_threads, eff);
  }
  fclose(f);
}

static void usage() {
  printf(
      "usage: cc <graph.egr> [options]\n"
      "  --ver LIST       comma list of V0,V1,V2,V3,V4 or 'all' (default all)\n"
      "  --segments N     segments for V1/V3/V4; 'auto' = round(2|E|/|V|) (default auto)\n"
      "  --t1 N --t2 N    dispatch thresholds on in-segment degree (default 4, 32)\n"
      "  --block N        threads per block, multiple of 32 (default 256)\n"
      "  --runs N         timed runs per version (default 5)\n"
      "  --warmup N       untimed warm-up runs per version (default 1)\n"
      "  --csv FILE       append one row per timed run\n"
      "  --seg-csv FILE   per-segment counters (counter build only)\n"
      "  --tag TEXT       free-text label written to the csv (e.g. GPU name)\n"
      "  --no-verify      skip CPU reference comparison\n"
      "  --stats          print graph statistics and exit\n"
      "  --emulate        run kernels' logic sequentially on the CPU (no GPU needed)\n");
}

static Config parse(int argc, char** argv) {
  Config c;
  if (argc < 2) { usage(); exit(1); }
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value"); return argv[++i]; };
    if (a == "--help" || a == "-h") { usage(); exit(0); }
    else if (a == "--ver") {
      std::string l = next(); c.versions.clear();
      if (l == "all") { c.versions = {V0, V1, V2, V3, V4}; continue; }
      size_t p = 0;
      while (p <= l.size()) {
        size_t q = l.find(',', p); if (q == std::string::npos) q = l.size();
        std::string t = l.substr(p, q - p); bool ok = false;
        for (int k = 0; k < NVER; k++) if (t == VNAME[k]) { c.versions.push_back((Ver)k); ok = true; }
        if (!ok) { fprintf(stderr, "unknown version %s\n", t.c_str()); exit(1); }
        p = q + 1;
      }
    }
    else if (a == "--segments") { std::string v = next(); c.segments = v == "auto" ? 0 : std::stoi(v); }
    else if (a == "--t1") c.t1 = std::stoi(next());
    else if (a == "--t2") c.t2 = std::stoi(next());
    else if (a == "--block") c.block = std::stoi(next());
    else if (a == "--runs") c.runs = std::stoi(next());
    else if (a == "--warmup") c.warmup = std::stoi(next());
    else if (a == "--csv") c.csv = next();
    else if (a == "--seg-csv") c.seg_csv = next();
    else if (a == "--tag") c.tag = next();
    else if (a == "--no-verify") c.verify = false;
    else if (a == "--stats") c.stats_only = true;
    else if (a == "--emulate") c.emulate = true;
    else if (a[0] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); exit(1); }
    else c.graph = a;
  }
  if (c.graph.empty()) die("no graph given");
  if (c.block % 32 || c.block < 32 || c.block > 1024) die("--block must be a multiple of 32 in [32,1024]");
  if (c.t1 < 0 || c.t2 < c.t1) die("need 0 <= t1 <= t2");
  if (c.segments < 0) die("--segments must be >= 1 or auto");
  if (c.runs < 1) die("--runs must be >= 1");
  return c;
}

int main(int argc, char** argv) {
  Config cfg = parse(argc, argv);
  std::string gname = base_name(cfg.graph);

  auto t0 = std::chrono::steady_clock::now();
  HostGraph hg = read_egr(cfg.graph);
  ForwardCSR f = make_forward(hg);
  hg = HostGraph();  // free
  double prep = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::vector<int> ref;
  if (cfg.verify || cfg.stats_only) ref = cpu_reference_labels(f);

  if (cfg.stats_only) {
    GraphStats s = compute_stats(f, ref);
    printf("graph,n,m,avg_deg,max_deg,deg_std,deg_cv,isolated,components,largest_cc,max_fwd_deg,auto_segments\n");
    printf("%s,%d,%lld,%.4f,%d,%.4f,%.4f,%lld,%lld,%lld,%d,%d\n", gname.c_str(), s.n, s.m, s.avg_deg,
           s.max_deg, s.deg_std, s.deg_cv, s.isolated, s.components, s.largest_cc, s.max_fwd_deg,
           auto_segments(f));
    return 0;
  }

  int S = cfg.segments ? cfg.segments : auto_segments(f);
  printf("graph %s: |V|=%d |E|=%lld  segments=%d%s  t1=%d t2=%d block=%d  (load+preprocess %.0f ms)\n",
         gname.c_str(), f.n, f.m, S, cfg.segments ? "" : " (auto)", cfg.t1, cfg.t2, cfg.block, prep);
  if (cfg.emulate) printf("CPU EMULATION MODE — times are NOT GPU times\n");

  DeviceGraph dg;
  if (!cfg.emulate) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d, %.1f GB)\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9);
    if (cfg.tag.empty()) cfg.tag = prop.name;
    bool need_light = false;
    for (Ver v : cfg.versions) need_light |= (v == V4);
    dg.upload(f, need_light);
  } else if (cfg.tag.empty()) cfg.tag = "emulated";

  bool all_ok = true;
  printf("%-4s %4s %10s %10s %10s %10s  %s\n", "ver", "s", "median_ms", "setup_ms", "hook_ms", "jump_ms", "correct");
  for (Ver v : cfg.versions) {
    int s = segmented(v) ? S : 1;
    std::vector<double> tot, setup, hook, jump;
    bool ok = true;
    for (int run = -cfg.warmup; run < cfg.runs; run++) {
      bool want_seg = (run == 0);
      RunResult r = cfg.emulate ? run_emulated(f, v, s, cfg) : run_gpu(dg, v, s, cfg, want_seg);
      int correct = -1;
      if (cfg.verify) { correct = same_partition(ref, r.labels) ? 1 : 0; ok &= (correct == 1); }
      if (run < 0) continue;  // warm-up
      csv_row(cfg, gname, v, run, r, correct, count_components(r.labels));
      if (want_seg) seg_csv_rows(cfg, gname, v, r);
      tot.push_back(r.total_ms); setup.push_back(r.setup_ms); hook.push_back(r.hook_ms); jump.push_back(r.jump_ms);
    }
    auto med = [](std::vector<double> x) { std::sort(x.begin(), x.end()); size_t n = x.size();
      return n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]); };
    printf("%-4s %4d %10.3f %10.3f %10.3f %10.3f  %s\n", VNAME[v], s, med(tot), med(setup), med(hook),
           med(jump), cfg.verify ? (ok ? "yes" : "NO  <-- WRONG PARTITION") : "skipped");
    all_ok &= ok;
  }
  if (!cfg.emulate) dg.release();
  return all_ok ? 0 : 3;
}
