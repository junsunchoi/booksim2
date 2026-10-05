// $Id$

/*collectivetrafficmanager.hpp
 *
 *Trace-driven traffic manager for collective schedules (sim_type = collective).
 *
 *trace_file lists one message per line (GPU ids are router ids):
 *  id src dst size_flits launch_cycle dim_order tie_mask ndeps [dep_id ...]
 *A message becomes eligible at max(launch_cycle, finish of its last dependency)
 *and is split into packets of at most packet_size flits.
 *On clos (GPU ids are GPU indices) a message's packets are striped over the
 *GPU's clos_gpu_ports terminals: each terminal takes the next packet and
 *sends it to the same rail of the destination GPU.
 *
 *trace_out writes one CSV row per message; trace_packets_out writes one row
 *per packet (creation, head injection and tail arrival times).
 *
 */

#ifndef _COLLECTIVETRAFFICMANAGER_HPP_
#define _COLLECTIVETRAFFICMANAGER_HPP_

#include <iostream>
#include <fstream>
#include <map>
#include <deque>
#include <queue>
#include <functional>

#include "config_utils.hpp"
#include "trafficmanager.hpp"

class CollectiveTrafficManager : public TrafficManager {

  struct Message {
    int id;
    int src, dst;
    int size;
    int launch;
    int dim_order;
    int tie_mask;
    int src_term, dst_term;         // clos: rail-0 terminals
    int pkts_total;
    vector<int> deps;
    vector<int> succs;

    int deps_left;
    int eligible;
    int first_inject;
    int finish;
    int pkts_sent;
    int pkts_done;
    int flits_sent;
  };

  typedef priority_queue<pair<int, int>, vector<pair<int, int> >,
                         greater<pair<int, int> > > tPendingQueue;

  struct PacketHead {
    int id;
    int ctime;
    int itime;
  };

  vector<Message> _msgs;
  vector<tPendingQueue> _pending;   // [terminal] (eligible time, message)
  vector<deque<int> > _active;      // [terminal] eligible messages with packets left
  vector<int> _next_msg;            // [terminal] message chosen by _IssuePacket

  int _gpus;
  int _terms_per_gpu;
  bool _hyperx;
  bool _torus;                      // topology = multilinktorus
  bool _stripe;                     // topology = clos: packets striped over rails
  bool _round_robin;
  int _max_packet_size;             // payload flits per packet
  int _header_flits;                // overhead flits per packet
  int _max_cycles;
  string _trace_out;
  string _trace_packets_out;
  ofstream * _packets_out;
  map<int, PacketHead> _packet_heads;  // [pid] head of packets in flight

  int _released;
  int _done;
  int _start_time;
  int _completion_time;

  void _LoadTrace( const string & filename );
  void _ResetState( );
  void _Release( int m, int time );
  void _Complete( int m, int time );
  void _WriteTraceOut( ) const;
  void _ReportLinkLoad( const vector<long long> & before,
                        const vector<long long> & inject_before,
                        const vector<long long> & eject_before ) const;
  vector<long long> _ChannelFlits( ) const;
  static vector<long long> _Flits( const vector<FlitChannel *> & chans );

  string _timeline_out;
  int _timeline_window;

protected:

  virtual int  _IssuePacket( int source, int cl );
  virtual void _GeneratePacket( int source, int stype, int cl, int time );
  virtual void _RetireFlit( Flit *f, int dest );
  virtual bool _SingleSim( );

public:

  CollectiveTrafficManager( const Configuration &config, const vector<Network *> & net );
  virtual ~CollectiveTrafficManager( );

  virtual void DisplayStats( ostream & os = cout ) const;
};

#endif
