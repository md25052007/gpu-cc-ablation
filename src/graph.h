// graph.h — host-side graph I/O, preprocessing and CPU reference.
//
// On-disk format: ECL-CC's binary CSR (".egr", see ECLgraph.h):
//   int32 nodes, int32 edges, int32 nindex[nodes+1], int32 nlist[edges]
//   (optional edge weights after that are ignored).
// Using the same format means the official ECL-CC binary can run on the
// exact same files we benchmark, which keeps the external comparison fair.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

struct HostGraph {          // graph exactly as stored in the .egr file
  int nodes = 0;
  int edges = 0;            // number of stored (directed) entries
  std::vector<int> nindex;  // nodes + 1
  std::vector<int> nlist;   // edges
};

// Forward CSR: every undirected edge {u,v} (u != v) is stored exactly once,
// in the row of min(u,v), pointing to max(u,v). Rows are sorted and
// deduplicated. All versions (V0..V4) run on this same structure, so
// "degree" means forward degree everywhere.
struct ForwardCSR {
  int n = 0;
  long long m = 0;          // undirected edge count (= forward entries)
  std::vector<int> off;     // n + 1
  std::vector<int> adj;     // m
};

static inline void die(const char* msg) {
  fprintf(stderr, "error: %s\n", msg);
  exit(1);
}

inline HostGraph read_egr(const std::string& path) {
  HostGraph g;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  if (fread(&g.nodes, sizeof(int), 1, f) != 1) die("bad header (nodes)");
  if (fread(&g.edges, sizeof(int), 1, f) != 1) die("bad header (edges)");
  if (g.nodes < 1 || g.edges < 0) die("bad header values");
  g.nindex.resize((size_t)g.nodes + 1);
  g.nlist.resize((size_t)g.edges);
  if (fread(g.nindex.data(), sizeof(int), g.nindex.size(), f) != g.nindex.size()) die("truncated nindex");
  if (g.edges > 0 &&
      fread(g.nlist.data(), sizeof(int), g.nlist.size(), f) != g.nlist.size()) die("truncated nlist");
  fclose(f);
  if (g.nindex[0] != 0 || g.nindex[g.nodes] != g.edges) die("nindex inconsistent with edge count");
  for (int v = 0; v < g.nodes; v++)
    if (g.nindex[v] > g.nindex[v + 1]) die("nindex not monotone");
  for (int x : g.nlist)
    if (x < 0 || x >= g.nodes) die("neighbor id out of range");
  return g;
}

// Builds the forward CSR. Robust to: one-directional storage, duplicate
// entries and self-loops (all of which appear in raw downloaded graphs).
inline ForwardCSR make_forward(const HostGraph& g) {
  ForwardCSR f;
  f.n = g.nodes;
  std::vector<long long> cnt((size_t)g.nodes + 1, 0);
  for (int u = 0; u < g.nodes; u++)
    for (int j = g.nindex[u]; j < g.nindex[u + 1]; j++) {
      int v = g.nlist[j];
      if (u != v) cnt[std::min(u, v)]++;
    }
  std::vector<long long> pos((size_t)g.nodes + 1, 0);
  for (int v = 0; v < g.nodes; v++) pos[v + 1] = pos[v] + cnt[v];
  std::vector<int> tmp((size_t)pos[g.nodes]);
  std::vector<long long> fill(pos.begin(), pos.end() - 1);
  for (int u = 0; u < g.nodes; u++)
    for (int j = g.nindex[u]; j < g.nindex[u + 1]; j++) {
      int v = g.nlist[j];
      if (u != v) tmp[fill[std::min(u, v)]++] = std::max(u, v);
    }
  // sort + unique each row, compact into f.adj
  f.off.assign((size_t)g.nodes + 1, 0);
  f.adj.reserve(tmp.size());
  for (int v = 0; v < g.nodes; v++) {
    auto b = tmp.begin() + pos[v], e = tmp.begin() + pos[v + 1];
    std::sort(b, e);
    auto ue = std::unique(b, e);
    f.adj.insert(f.adj.end(), b, ue);
    if (f.adj.size() > (size_t)INT32_MAX) die("graph too large for 32-bit offsets");
    f.off[v + 1] = (int)f.adj.size();
  }
  f.m = (long long)f.adj.size();
  return f;
}

// ---------- CPU reference: sequential union-find (path halving + union by size)
inline std::vector<int> cpu_reference_labels(const ForwardCSR& f) {
  std::vector<int> p(f.n), sz(f.n, 1);
  std::iota(p.begin(), p.end(), 0);
  auto find = [&](int x) {
    while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
  };
  for (int u = 0; u < f.n; u++)
    for (int j = f.off[u]; j < f.off[u + 1]; j++) {
      int a = find(u), b = find(f.adj[j]);
      if (a == b) continue;
      if (sz[a] < sz[b]) std::swap(a, b);
      p[b] = a; sz[a] += sz[b];
    }
  for (int v = 0; v < f.n; v++) p[v] = find(v);
  return p;
}

// Two labelings describe the same partition iff the label mapping is a
// bijection between their classes. Label VALUES do not need to match.
inline bool same_partition(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) return false;
  size_t n = a.size();
  std::vector<int> ab(n, -1), ba(n, -1);
  for (size_t v = 0; v < n; v++) {
    int x = a[v], y = b[v];
    if (x < 0 || (size_t)x >= n || y < 0 || (size_t)y >= n) return false;
    if (ab[x] == -1) ab[x] = y; else if (ab[x] != y) return false;
    if (ba[y] == -1) ba[y] = x; else if (ba[y] != x) return false;
  }
  return true;
}

inline long long count_components(const std::vector<int>& labels) {
  long long c = 0;
  for (size_t v = 0; v < labels.size(); v++) c += (labels[v] == (int)v);
  return c;
}

// ---------- Graph statistics for the dataset metadata table
struct GraphStats {
  int n; long long m; double avg_deg; int max_deg; double deg_std; double deg_cv;
  long long isolated; long long components; long long largest_cc;
  int max_fwd_deg;
};

inline GraphStats compute_stats(const ForwardCSR& f, const std::vector<int>& ref) {
  GraphStats s{};
  s.n = f.n; s.m = f.m;
  std::vector<int> deg(f.n, 0);
  for (int u = 0; u < f.n; u++) {
    int d = f.off[u + 1] - f.off[u];
    s.max_fwd_deg = std::max(s.max_fwd_deg, d);
    deg[u] += d;
    for (int j = f.off[u]; j < f.off[u + 1]; j++) deg[f.adj[j]]++;
  }
  double mean = 2.0 * f.m / f.n, var = 0;
  for (int d : deg) {
    s.max_deg = std::max(s.max_deg, d);
    var += (d - mean) * (d - mean);
    s.isolated += (d == 0);
  }
  s.avg_deg = mean;
  s.deg_std = std::sqrt(var / f.n);
  s.deg_cv = mean > 0 ? s.deg_std / mean : 0;
  std::vector<long long> csize(f.n, 0);
  for (int v = 0; v < f.n; v++) csize[ref[v]]++;
  for (int v = 0; v < f.n; v++) {
    if (ref[v] == v) s.components++;
    s.largest_cc = std::max(s.largest_cc, csize[v]);
  }
  return s;
}
