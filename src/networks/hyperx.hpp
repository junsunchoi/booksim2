// $Id$

/*hyperx.hpp
 *
 *HyperX (flattened butterfly) with an independent size per dimension: each
 *router connects to every other router in each of its dimensions.
 *For a torus see multilinktorus.hpp.
 *
 *Each router has c terminals (default: one per network port) so that a GPU
 *can inject into and eject from every link concurrently.
 *
 *Port layout per router: network ports first, grouped by dimension
 *(offset 1..L-1), then the c terminal ports.
 */

#ifndef _HYPERX_HPP_
#define _HYPERX_HPP_

#include "network.hpp"

class HyperX : public Network {

  void _ComputeSize( const Configuration &config );
  void _BuildNet( const Configuration &config );

public:
  HyperX( const Configuration &config, const string & name );
  static void RegisterRoutingFunctions();
};

namespace hyperx {

  int NumDims();
  int DimSize(int dim);
  int NumPorts();
  int Terminals();
  int MaxHops();

  int Coord(int router, int dim);
  int Neighbor(int router, int port);
  int ReversePort(int port);

  // Next network output port on the dimension-ordered path from cur to dest,
  // or -1 if cur == dest. dim_order is the index of the first dimension
  // (dimensions are visited in rotation order). tie_mask is unused (every
  // dimension is a single hop) and kept for interface parity with
  // multilinktorus.
  int NextPort(int cur, int dest, int dim_order, int tie_mask);

  int FirstHopPort(int src, int dest, int dim_order, int tie_mask);
  // Input port at dest through which the packet arrives.
  int ArrivalPort(int src, int dest, int dim_order, int tie_mask);
  int PathHops(int src, int dest);
}

#endif
