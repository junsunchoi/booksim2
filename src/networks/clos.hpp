// $Id$

/*clos.hpp
 *
 *Rail-optimized Clos built from radix-R switches. Each GPU has P ports
 *(clos_gpu_ports, one terminal each): terminal g * P + r is port (rail) r of
 *GPU g.
 *  clos_levels = 1: P rail switches; port r of every GPU connects to switch r
 *                   (clos_gpus <= R), e.g. 64 GPUs x 16 ports -> 16 switches.
 *  clos_levels = 2: pods of R/2 GPUs. Each pod has P L1 switches (one per
 *                   rail) with R/2 GPU ports and R/2 up ports. Rails are
 *                   grouped G = min(P, R / pods) at a time; rail group k has
 *                   R/2 L2 switches, and up port u of L1 switch (pod p, rail r)
 *                   with r / G = k connects to L2 switch k * R/2 + u. E.g. 256
 *                   GPUs, P = 16, R = 64: 8 pods x 16 L1 switches, 2 rail
 *                   groups (rails 0-7, 8-15) x 32 L2 switches with 64 ports.
 *
 *Router ids: L1 switch (pod p, rail r) is p * P + r, then the L2 switches.
 *L1 port layout: GPU ports 0..D-1 (local GPU index within the pod), then up
 *port u at D + u. L2 port p * G + r % G connects to L1 switch (pod p, rail r).
 */

#ifndef _CLOS_HPP_
#define _CLOS_HPP_

#include "network.hpp"

class Clos : public Network {

  void _ComputeSize( const Configuration &config );
  void _BuildNet( const Configuration &config );

public:
  Clos( const Configuration &config, const string & name );
  static void RegisterRoutingFunctions();
};

namespace clos {

  // Terminals (ports, rails) per GPU.
  int Terminals();
}

#endif
