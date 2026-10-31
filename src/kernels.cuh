// kernels.cuh — the building blocks shared by every version.
//
//   Atomic-Hook  : lock-free CAS merge of two component trees
//   Multi-Jump   : fused pointer jumping until every tree is a star
//   Slicing      : per-vertex slicing of the neighbor list into s segments
//   Dispatch     : thread / warp / block hook kernels
//
// Invariant used for correctness: parent[x] <= x for every x. Hook always
// attaches the LARGER root under the SMALLER one, and Multi-Jump only moves
// pointers further up the same tree, so pointers only ever decrease and no
// cycle can form.
//
// Each per-thread body is __host__ __device__ so the exact same code can be
// run sequentially on the CPU (--emulate). That checks indexing, slicing and
// worklist logic without a GPU; it does NOT test concurrency (only real GPU
// runs do).
#pragma once
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu

// ---------------------------------------------------------------- counters
struct DevCounters {
  unsigned long long cas_attempts;  // atomicCAS calls inside Atomic-Hook
  unsigned long long cas_retries;   // CAS calls that lost a race
  unsigned long long find_steps;    // parent-pointer hops while finding roots
  unsigned long long lane_work;     // sum over lanes of edges processed
  unsigned long long lane_slots;    // sum over busy warps of 32 * max lane work
  unsigned long long busy_lanes;    // lanes that processed >= 1 edge
  unsigned long long max_lane_work; // largest edge count of any single lane
  unsigned long long idle_threads;  // launched threads that had no edge at all
};

#ifdef CC_COUNTERS
__device__ DevCounters d_ctr;
#endif

struct LocalCtr {
  unsigned long long att = 0, ret = 0, steps = 0;
};

// ---------------------------------------------------------------- atomics
__host__ __device__ __forceinline__ int cas_int(int* addr, int cmp, int val) {
#ifdef __CUDA_ARCH__
  return atomicCAS(addr, cmp, val);
#else
  int old = *addr;               // CPU emulation is sequential
  if (old == cmp) *addr = val;
  return old;
#endif
}

__host__ __device__ __forceinline__ int find_root(int* parent, int v, LocalCtr& c) {
  volatile int* p = parent;      // always re-read memory: other threads hook concurrently
  int x = p[v];
  while (x != v) { v = x; x = p[v]; c.steps++; }
  return v;
}

// Atomic-Hook (Sutton et al. 2016): find both roots, try to attach the larger
// root under the smaller with CAS; if another thread changed that root first,
// re-find and retry. Terminates once both endpoints share a root.
__host__ __device__ __forceinline__ void atomic_hook(int* parent, int u, int v, LocalCtr& c) {
  while (true) {
    u = find_root(parent, u, c);
    v = find_root(parent, v, c);
    if (u == v) return;
    int hi = u > v ? u : v;
    int lo = u > v ? v : u;
    c.att++;
    if (cas_int(&parent[hi], hi, lo) == hi) return;
    c.ret++;
    u = hi; v = lo;
  }
}

// ---------------------------------------------------------------- Slicing
// Segment k of s for vertex v covers neighbor positions [b, e):
//   b = off[v] + floor(k * d / s),  e = off[v] + floor((k+1) * d / s)
// Over k = 0..s-1 the slices partition the row exactly; each has d/s +- 1 edges.
__host__ __device__ __forceinline__ void slice_of(const int* off, int v, int k, int s,
                                                  int& b, int& e) {
  int o = off[v];
  long long d = (long long)off[v + 1] - o;
  b = o + (int)((d * k) / s);
  e = o + (int)((d * (k + 1)) / s);
}

// Bucket by in-segment degree ceil(d/s). With s = 1 this is the full degree,
// i.e. plain ECL-CC-style dispatch. 0 = thread, 1 = warp, 2 = block.
// Zero-degree vertices go to the thread bucket on purpose, so V2/V3 launch
// for exactly the same vertex set as V0/V1 and dispatch is the only change.
__host__ __device__ __forceinline__ int bucket_of(int d, int s, int t1, int t2) {
  int ds = (d + s - 1) / s;
  return ds <= t1 ? 0 : (ds <= t2 ? 1 : 2);
}

// Hook neighbors j = b+start, b+start+stride, ... < e of vertex v.
// Thread: (0,1). Warp: (lane,32). Block: (threadIdx,blockDim).
__host__ __device__ __forceinline__ int hook_range(const int* adj, int* parent, int v,
                                                   int b, int e, int start, int stride,
                                                   LocalCtr& c) {
  int w = 0;
  for (int j = b + start; j < e; j += stride) { atomic_hook(parent, v, adj[j], c); w++; }
  return w;
}

// Multi-Jump body (Sutton et al.): while parent(parent(v)) != parent(v),
// set parent(v) <- parent(parent(v)). Writes every step so other threads
// walking through v benefit immediately.
__host__ __device__ __forceinline__ void multijump_vertex(int* parent, int v) {
  volatile int* p = parent;
  int a = p[v];
  int b = p[a];
  while (a != b) { p[v] = b; a = b; b = p[a]; }
}

// ---------------------------------------------------------------- GPU only
#ifdef __CUDACC__

// Flush per-thread counters + per-warp load-balance statistics.
// Must be reached by ALL 32 lanes of the warp (no early returns in kernels).
__device__ __forceinline__ void flush_counters(const LocalCtr& c, int w) {
#ifdef CC_COUNTERS
  unsigned long long att = c.att, ret = c.ret, st = c.steps;
  unsigned long long ws = w, wm = w, busy = (w > 0), idle = (w == 0);
  for (int o = 16; o > 0; o >>= 1) {
    att += __shfl_xor_sync(FULL_MASK, att, o);
    ret += __shfl_xor_sync(FULL_MASK, ret, o);
    st += __shfl_xor_sync(FULL_MASK, st, o);
    ws += __shfl_xor_sync(FULL_MASK, ws, o);
    busy += __shfl_xor_sync(FULL_MASK, busy, o);
    idle += __shfl_xor_sync(FULL_MASK, idle, o);
    unsigned long long other = __shfl_xor_sync(FULL_MASK, wm, o);
    wm = other > wm ? other : wm;
  }
  if ((threadIdx.x & 31) == 0) {
    if (att) atomicAdd(&d_ctr.cas_attempts, att);
    if (ret) atomicAdd(&d_ctr.cas_retries, ret);
    if (st) atomicAdd(&d_ctr.find_steps, st);
    if (idle) atomicAdd(&d_ctr.idle_threads, idle);
    if (wm > 0) {
      atomicAdd(&d_ctr.lane_work, ws);
      atomicAdd(&d_ctr.lane_slots, 32ull * wm);
      atomicAdd(&d_ctr.busy_lanes, busy);
      atomicMax(&d_ctr.max_lane_work, wm);
    }
  }
#else
  (void)c; (void)w;
#endif
}

__global__ void k_init(int n, int* parent) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) parent[v] = v;
}

__global__ void k_multijump(int n, int* parent) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) multijump_vertex(parent, v);
}

// V0 / V1: one thread per vertex, every vertex, slice k of s.
__global__ void k_hook_vertex_all(int n, const int* __restrict__ off,
                                  const int* __restrict__ adj, int* parent, int k, int s) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  LocalCtr c; int w = 0;
  if (v < n) { int b, e; slice_of(off, v, k, s, b, e); w = hook_range(adj, parent, v, b, e, 0, 1, c); }
  flush_counters(c, w);
}

// V2 / V3 thread bucket: one thread per listed vertex.
__global__ void k_hook_thread(int cnt, const int* __restrict__ wl, const int* __restrict__ off,
                              const int* __restrict__ adj, int* parent, int k, int s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  LocalCtr c; int w = 0;
  if (i < cnt) { int v = wl[i], b, e; slice_of(off, v, k, s, b, e); w = hook_range(adj, parent, v, b, e, 0, 1, c); }
  flush_counters(c, w);
}

// V2 / V3 warp bucket: one warp per listed vertex, lanes stride by 32.
__global__ void k_hook_warp(int cnt, const int* __restrict__ wl, const int* __restrict__ off,
                            const int* __restrict__ adj, int* parent, int k, int s) {
  long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int i = (int)(gt >> 5), lane = threadIdx.x & 31;
  LocalCtr c; int w = 0;
  if (i < cnt) { int v = wl[i], b, e; slice_of(off, v, k, s, b, e); w = hook_range(adj, parent, v, b, e, lane, 32, c); }
  flush_counters(c, w);
}

// V2 / V3 block bucket: one block per listed vertex, threads stride by blockDim.
__global__ void k_hook_block(const int* __restrict__ wl, const int* __restrict__ off,
                             const int* __restrict__ adj, int* parent, int k, int s) {
  int v = wl[blockIdx.x], b, e;
  slice_of(off, v, k, s, b, e);
  LocalCtr c;
  int w = hook_range(adj, parent, v, b, e, threadIdx.x, blockDim.x, c);
  flush_counters(c, w);
}

// ---------------------------------------------------------------- V4 setup
// V4 splits vertices by in-segment degree, exactly like V3's buckets:
//   light (bucket 0)      -> its edges go into a flat edge list, hooked one thread per edge
//                            in s contiguous chunks (no idle threads)
//   heavy (buckets 1 / 2) -> sliced neighbour lists with warp / block dispatch (V3-style)
// light_deg(v) is the number of entries v contributes to the flat list.
struct LightDeg {
  const int* off; int s, t1, t2;
  __host__ __device__ int operator()(int v) const {
    int d = off[v + 1] - off[v];
    return bucket_of(d, s, t1, t2) == 0 ? d : 0;
  }
};

// Copies light edges into (lsrc, ldst) at offsets pos[] (exclusive scan of light_deg).
// 8 lanes per vertex: light vertices have at most t1*s edges, so this keeps the copy
// reasonably coalesced without wasting a full warp on degree-2 vertices.
#define LIGHT_GROUP 8
__global__ void k_build_light(int n, const int* __restrict__ off, const int* __restrict__ adj,
                              const int* __restrict__ pos, int* lsrc, int* ldst) {
  long long gt = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long v = gt / LIGHT_GROUP;
  int lane = (int)(gt % LIGHT_GROUP);
  if (v >= n) return;
  int p = pos[v], cnt = pos[v + 1] - p, o = off[v];
  for (int i = lane; i < cnt; i += LIGHT_GROUP) { lsrc[p + i] = (int)v; ldst[p + i] = adj[o + i]; }
}

// V4 light part: one thread per light edge, edges [eb, ee) of the flat light-edge list.
__global__ void k_hook_light(long long eb, long long ee, const int* __restrict__ src,
                            const int* __restrict__ adj, int* parent) {
  long long i = eb + (long long)blockIdx.x * blockDim.x + threadIdx.x;
  LocalCtr c; int w = 0;
  if (i < ee) { atomic_hook(parent, src[i], adj[i], c); w = 1; }
  flush_counters(c, w);
}

#endif  // __CUDACC__
