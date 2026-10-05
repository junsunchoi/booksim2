// $Id$

/*hyperx.cpp
 *
 *HyperX (all-to-all per dimension) with arbitrary dimension sizes, plus the
 *dimension-order routing function dor_hyperx.
 *
 */

#include "booksim.hpp"
#include <algorithm>
#include <vector>
#include <sstream>
#include <cassert>
#include <cstdlib>

#include "hyperx.hpp"
#include "routefunc.hpp"
#include "misc_utils.hpp"

namespace {

  vector<int> gDimSize;
  vector<int> gDimStride;
  vector<int> gDimPortBase;
  int gPorts = 0;
  int gTerminals = 0;
  int gMaxHops = 0;
  bool gSingleHopAllVCs = true;

  int _DimPorts(int d)
  {
    return gDimSize[d] - 1;
  }

  int _PortDim(int port)
  {
    int d = (int)gDimPortBase.size() - 1;
    while(gDimPortBase[d] > port) {
      --d;
    }
    return d;
  }

}

namespace hyperx {

  int NumDims() { return gDimSize.size(); }
  int DimSize(int dim) { return gDimSize[dim]; }
  int NumPorts() { return gPorts; }
  int Terminals() { return gTerminals; }
  int MaxHops() { return gMaxHops; }

  int Coord(int router, int dim)
  {
    return (router / gDimStride[dim]) % gDimSize[dim];
  }

  int Neighbor(int router, int port)
  {
    assert((port >= 0) && (port < gPorts));
    int const d = _PortDim(port);
    int const L = gDimSize[d];
    int const off = port - gDimPortBase[d];
    int const c = Coord(router, d);
    int const step = off + 1;
    int const nc = (c + step) % L;
    return router + (nc - c) * gDimStride[d];
  }

  int ReversePort(int port)
  {
    int const d = _PortDim(port);
    int const L = gDimSize[d];
    int const off = port - gDimPortBase[d];
    return gDimPortBase[d] + (L - (off + 1)) - 1;
  }

  int NextPort(int cur, int dest, int dim_order, int tie_mask)
  {
    int const D = gDimSize.size();
    int const start = ((dim_order % D) + D) % D;
    for(int i = 0; i < D; ++i) {
      int const d = (start + i) % D;
      int const L = gDimSize[d];
      int const cc = Coord(cur, d);
      int const dc = Coord(dest, d);
      if(cc == dc) {
        continue;
      }
      int const fwd = (dc - cc + L) % L;
      return gDimPortBase[d] + fwd - 1;
    }
    return -1;
  }

  int FirstHopPort(int src, int dest, int dim_order, int tie_mask)
  {
    return NextPort(src, dest, dim_order, tie_mask);
  }

  int ArrivalPort(int src, int dest, int dim_order, int tie_mask)
  {
    int cur = src;
    int last = -1;
    int port;
    while((port = NextPort(cur, dest, dim_order, tie_mask)) >= 0) {
      last = port;
      cur = Neighbor(cur, port);
    }
    return (last < 0) ? -1 : ReversePort(last);
  }

  int PathHops(int src, int dest)
  {
    int hops = 0;
    for(size_t d = 0; d < gDimSize.size(); ++d) {
      if(Coord(dest, d) != Coord(src, d)) {
        ++hops;
      }
    }
    return hops;
  }

}

HyperX::HyperX( const Configuration &config, const string & name ) :
  Network( config, name )
{
  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );
}

void HyperX::_ComputeSize( const Configuration &config )
{
  gDimSize = config.GetIntArray( "dim_sizes" );
  if(gDimSize.empty()) {
    Error( "hyperx: dim_sizes must be set, e.g. dim_sizes = {4,4,4};" );
  }
  int const D = gDimSize.size();

  gDimStride.resize(D);
  gDimPortBase.resize(D);
  _size = 1;
  gPorts = 0;
  gMaxHops = 0;
  for(int d = 0; d < D; ++d) {
    if(gDimSize[d] < 1) {
      Error( "hyperx: dimension sizes must be positive" );
    }
    gDimStride[d] = _size;
    _size *= gDimSize[d];
    gDimPortBase[d] = gPorts;
    gPorts += _DimPorts(d);
    if(gDimSize[d] > 1) {
      ++gMaxHops;
    }
  }

  gTerminals = config.GetInt( "hyperx_c" );
  if(gTerminals <= 0) {
    gTerminals = gPorts;
  }
  gSingleHopAllVCs = (config.GetInt( "hyperx_single_hop_all_vcs" ) != 0);

  gN = D;
  gK = 0;
  for(int d = 0; d < D; ++d) {
    gK = max(gK, gDimSize[d]);
  }
  gC = gTerminals;

  _channels = _size * gPorts;
  _nodes = _size * gTerminals;
}

void HyperX::_BuildNet( const Configuration &config )
{
  int const latency = config.GetInt( "hyperx_link_latency" );
  int const D = gDimSize.size();

  for(int r = 0; r < _size; ++r) {

    ostringstream router_name;
    router_name << "router";
    for(int d = 0; d < D; ++d) {
      router_name << "_" << hyperx::Coord(r, d);
    }

    _routers[r] = Router::NewRouter( config, this, router_name.str( ),
                                     r, gPorts + gTerminals, gPorts + gTerminals );
    _timed_modules.push_back(_routers[r]);

    // Input port q receives from Neighbor(r, q), which sends on its port ReversePort(q).
    for(int q = 0; q < gPorts; ++q) {
      int const ch = hyperx::Neighbor(r, q) * gPorts + hyperx::ReversePort(q);
      _routers[r]->AddInputChannel( _chan[ch], _chan_cred[ch] );
    }
    for(int t = 0; t < gTerminals; ++t) {
      int const node = r * gTerminals + t;
      _routers[r]->AddInputChannel( _inject[node], _inject_cred[node] );
      _inject[node]->SetLatency( 1 );
      _inject_cred[node]->SetLatency( 1 );
    }

    for(int p = 0; p < gPorts; ++p) {
      int const ch = r * gPorts + p;
      _routers[r]->AddOutputChannel( _chan[ch], _chan_cred[ch] );
      _chan[ch]->SetLatency( latency );
      _chan_cred[ch]->SetLatency( latency );
    }
    for(int t = 0; t < gTerminals; ++t) {
      int const node = r * gTerminals + t;
      _routers[r]->AddOutputChannel( _eject[node], _eject_cred[node] );
      _eject[node]->SetLatency( 1 );
      _eject_cred[node]->SetLatency( 1 );
    }
  }
}

// Dimension-order routing. Each flit carries its own dimension rotation
// (f->dim_order) and L/2 tie directions (f->tie_dir bitmask). VCs are split
// into MaxHops() classes and a packet uses class h on its (h+1)-th network
// hop, which is deadlock-free for any mix of dimension orders. Packets whose
// whole path is a single hop may use every VC.
void dor_hyperx( const Router *r, const Flit *f, int in_channel,
                 OutputSet *outputs, bool inject )
{
  outputs->Clear( );

  if(inject) {
    outputs->AddRange( -1, 0, gNumVCs - 1 );
    return;
  }

  int const cur = r->GetID( );
  int const dest = f->dest / gTerminals;

  if(cur == dest) {
    outputs->AddRange( gPorts + (f->dest % gTerminals), 0, gNumVCs - 1 );
    return;
  }

  int const out_port = hyperx::NextPort( cur, dest, f->dim_order, f->tie_dir );
  assert(out_port >= 0);

  if(gSingleHopAllVCs && (hyperx::PathHops( f->src / gTerminals, dest ) == 1)) {
    outputs->AddRange( out_port, 0, gNumVCs - 1 );
    return;
  }

  int const vcs_per_class = gNumVCs / gMaxHops;
  if(vcs_per_class < 1) {
    ostringstream err;
    err << "dor_hyperx: num_vcs (" << gNumVCs << ") must be at least the maximum hop count ("
        << gMaxHops << ")";
    r->Error( err.str( ) );
  }
  int const vc_class = (in_channel >= gPorts) ? 0 : (f->vc / vcs_per_class + 1);
  assert(vc_class < gMaxHops);

  int const vc_begin = vc_class * vcs_per_class;
  outputs->AddRange( out_port, vc_begin, vc_begin + vcs_per_class - 1 );

  if(f->watch) {
    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
               << "dor_hyperx: flit " << f->id << " dest " << f->dest
               << " out port " << out_port << " VCs [" << vc_begin << ","
               << vc_begin + vcs_per_class - 1 << "]." << endl;
  }
}

void HyperX::RegisterRoutingFunctions()
{
  gRoutingFunctionMap["dor_hyperx"] = &dor_hyperx;
}
