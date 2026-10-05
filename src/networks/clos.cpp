// $Id$

/*clos.cpp
 *
 *One- or two-level rail-optimized Clos with up/down routing functions
 *random_clos, adaptive_clos and dest_clos (up-port selection policy).
 *
 */

#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cassert>

#include "clos.hpp"
#include "routefunc.hpp"
#include "random_utils.hpp"
#include "misc_utils.hpp"

namespace {

  int gGpus = 0;
  int gPorts = 0;      // ports (rails) per GPU
  int gRadix = 0;
  int gLevels = 0;
  int gPods = 0;
  int gDown = 0;       // GPU ports per L1 switch (GPUs per pod)
  int gUp = 0;         // up ports per L1 switch (= L2 switches per rail group)
  int gGroup = 0;      // rails per rail group
  int gL1 = 0;
  int gL2 = 0;
  int gL2Ports = 0;
  bool gAlwaysUp = false;

  int _UpChannel(int l1, int u)
  {
    return l1 * gUp + u;
  }

  int _DownChannel(int l2, int port)
  {
    return gL1 * gUp + l2 * gL2Ports + port;
  }

}

namespace clos {

  int Terminals()
  {
    return gPorts;
  }

}

Clos::Clos( const Configuration &config, const string & name ) :
  Network( config, name )
{
  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );
}

void Clos::_ComputeSize( const Configuration &config )
{
  gGpus = config.GetInt( "clos_gpus" );
  gPorts = config.GetInt( "clos_gpu_ports" );
  gRadix = config.GetInt( "clos_radix" );
  gLevels = config.GetInt( "clos_levels" );
  gAlwaysUp = (config.GetInt( "clos_always_up" ) != 0);
  if((gGpus < 2) || (gRadix < 2) || (gPorts < 1)) {
    Error( "clos: clos_gpus and clos_radix must be at least 2, clos_gpu_ports at least 1" );
  }
  if(gLevels <= 0) {
    gLevels = (gGpus <= gRadix) ? 1 : 2;
  }

  if(gLevels == 1) {
    if(gGpus > gRadix) {
      Error( "clos: one level needs clos_gpus <= clos_radix" );
    }
    gPods = 1;
    gDown = gGpus;
    gUp = 0;
    gGroup = gPorts;
    gL2 = 0;
    gL2Ports = 0;
  } else if(gLevels == 2) {
    gDown = gRadix / 2;
    gUp = gRadix - gDown;
    if((gRadix % 2) || (gGpus % gDown)) {
      Error( "clos: two levels need an even radix R and clos_gpus a multiple of R/2" );
    }
    gPods = gGpus / gDown;
    if(gPods > gRadix) {
      Error( "clos: two levels support at most R * R/2 GPUs" );
    }
    gGroup = min(gPorts, gRadix / gPods);
    if(gPorts % gGroup) {
      Error( "clos: clos_gpu_ports must be a multiple of the rails per group, min(P, R / pods)" );
    }
    gL2 = (gPorts / gGroup) * gUp;
    gL2Ports = gPods * gGroup;
  } else {
    Error( "clos: clos_levels must be 1 or 2" );
  }
  gL1 = gPods * gPorts;

  gN = gLevels;
  gK = gRadix;
  gC = gPorts;

  _size = gL1 + gL2;
  _nodes = gGpus * gPorts;
  _channels = 2 * gL1 * gUp;

  cout << "Clos: " << gGpus << " GPUs x " << gPorts << " ports, radix " << gRadix << ", "
       << gLevels << " level(s), " << gPods << " pod(s) x " << gPorts << " L1 switches x "
       << gDown << " GPU ports";
  if(gLevels == 2) {
    cout << ", " << gPorts / gGroup << " rail group(s) of " << gGroup << " rails x " << gUp
         << " L2 switches x " << gL2Ports << " ports";
  }
  cout << "." << endl;
}

void Clos::_BuildNet( const Configuration &config )
{
  int const latency = config.GetInt( "clos_link_latency" );

  for(int l = 0; l < gL1; ++l) {
    int const p = l / gPorts;
    int const r = l % gPorts;
    ostringstream router_name;
    router_name << "l1_" << p << "_" << r;
    int const ports = gDown + gUp;
    _routers[l] = Router::NewRouter( config, this, router_name.str( ), l, ports, ports );
    _timed_modules.push_back(_routers[l]);

    for(int i = 0; i < gDown; ++i) {
      int const node = (p * gDown + i) * gPorts + r;
      _routers[l]->AddInputChannel( _inject[node], _inject_cred[node] );
      _inject[node]->SetLatency( 1 );
      _inject_cred[node]->SetLatency( 1 );
    }
    for(int u = 0; u < gUp; ++u) {
      int const ch = _DownChannel( (r / gGroup) * gUp + u, p * gGroup + r % gGroup );
      _routers[l]->AddInputChannel( _chan[ch], _chan_cred[ch] );
    }

    for(int i = 0; i < gDown; ++i) {
      int const node = (p * gDown + i) * gPorts + r;
      _routers[l]->AddOutputChannel( _eject[node], _eject_cred[node] );
      _eject[node]->SetLatency( 1 );
      _eject_cred[node]->SetLatency( 1 );
    }
    for(int u = 0; u < gUp; ++u) {
      int const ch = _UpChannel( l, u );
      _routers[l]->AddOutputChannel( _chan[ch], _chan_cred[ch] );
      _chan[ch]->SetLatency( latency );
      _chan_cred[ch]->SetLatency( latency );
    }
  }

  for(int s = 0; s < gL2; ++s) {
    int const k = s / gUp;
    int const u = s % gUp;
    ostringstream router_name;
    router_name << "l2_" << s;
    int const id = gL1 + s;
    _routers[id] = Router::NewRouter( config, this, router_name.str( ), id, gL2Ports, gL2Ports );
    _timed_modules.push_back(_routers[id]);

    for(int q = 0; q < gL2Ports; ++q) {
      int const l = (q / gGroup) * gPorts + k * gGroup + q % gGroup;
      int const ch = _UpChannel( l, u );
      _routers[id]->AddInputChannel( _chan[ch], _chan_cred[ch] );
    }
    for(int q = 0; q < gL2Ports; ++q) {
      int const ch = _DownChannel( s, q );
      _routers[id]->AddOutputChannel( _chan[ch], _chan_cred[ch] );
      _chan[ch]->SetLatency( latency );
      _chan_cred[ch]->SetLatency( latency );
    }
  }
}

namespace {

  enum UpPolicy { up_random, up_adaptive, up_dest };

  // Up/down routing on the destination terminal's rail. The down path is
  // fixed by the destination: L2 port (dest pod, dest rail), so a packet can
  // only change rails within a rail group. With the dest policy, up port u
  // carries traffic for local GPU index u only, so every L2 -> L1 link serves
  // exactly one destination terminal. With clos_always_up, same-L1 packets
  // also turn around at an L2 switch so every path is L1 -> L2 -> L1.
  void _RouteClos( const Router *r, const Flit *f, int in_channel,
                   OutputSet *outputs, bool inject, UpPolicy policy )
  {
    outputs->Clear( );

    if(inject) {
      outputs->AddRange( -1, 0, gNumVCs - 1 );
      return;
    }

    int const cur = r->GetID( );
    int const dest_gpu = f->dest / gPorts;
    int const dest_rail = f->dest % gPorts;
    int const dest_pod = dest_gpu / gDown;
    int const dest_idx = dest_gpu % gDown;
    int out_port;

    if(cur >= gL1) {
      assert(dest_rail / gGroup == (cur - gL1) / gUp);
      out_port = dest_pod * gGroup + dest_rail % gGroup;
    } else if((cur == dest_pod * gPorts + dest_rail) &&
              ((gLevels == 1) || !gAlwaysUp || (in_channel >= gDown))) {
      out_port = dest_idx;
    } else {
      assert((gLevels == 2) && (dest_rail / gGroup == (cur % gPorts) / gGroup));
      if(policy == up_dest) {
        out_port = gDown + dest_idx % gUp;
      } else if(policy == up_random) {
        out_port = gDown + RandomInt( gUp - 1 );
      } else {
        int best = -1;
        int ties = 0;
        for(int u = 0; u < gUp; ++u) {
          int const used = r->GetUsedCredit( gDown + u );
          if((best < 0) || (used < best)) {
            best = used;
            ties = 1;
            out_port = gDown + u;
          } else if((used == best) && (RandomInt( ties++ ) == 0)) {
            out_port = gDown + u;
          }
        }
      }
    }

    outputs->AddRange( out_port, 0, gNumVCs - 1 );

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
                 << "clos: flit " << f->id << " dest " << f->dest
                 << " out port " << out_port << "." << endl;
    }
  }

}

void random_clos( const Router *r, const Flit *f, int in_channel,
                  OutputSet *outputs, bool inject )
{
  _RouteClos( r, f, in_channel, outputs, inject, up_random );
}

void adaptive_clos( const Router *r, const Flit *f, int in_channel,
                    OutputSet *outputs, bool inject )
{
  _RouteClos( r, f, in_channel, outputs, inject, up_adaptive );
}

void dest_clos( const Router *r, const Flit *f, int in_channel,
                OutputSet *outputs, bool inject )
{
  _RouteClos( r, f, in_channel, outputs, inject, up_dest );
}

void Clos::RegisterRoutingFunctions()
{
  gRoutingFunctionMap["random_clos"] = &random_clos;
  gRoutingFunctionMap["adaptive_clos"] = &adaptive_clos;
  gRoutingFunctionMap["dest_clos"] = &dest_clos;
}
