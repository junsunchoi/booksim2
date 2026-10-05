// $Id$

/*collectivetrafficmanager.cpp
 *
 *Trace-driven traffic manager for collective schedules (sim_type = collective).
 *
 */

#include <cassert>
#include <climits>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

#include "booksim.hpp"
#include "collectivetrafficmanager.hpp"
#include "clos.hpp"
#include "hyperx.hpp"
#include "multilinktorus.hpp"
#include "random_utils.hpp"

CollectiveTrafficManager::CollectiveTrafficManager( const Configuration &config,
                                                    const vector<Network *> & net )
  : TrafficManager(config, net), _packets_out(0), _released(0), _done(0),
    _start_time(0), _completion_time(-1)
{
  if(_classes != 1) {
    Error( "collective: only classes = 1 is supported" );
  }

  string const topo = config.GetStr( "topology" );
  _hyperx = (topo == "hyperx");
  _torus = (topo == "multilinktorus");
  _stripe = (topo == "clos");
  _terms_per_gpu = _hyperx ? hyperx::Terminals()
                 : _torus  ? multilinktorus::Terminals()
                 : _stripe ? clos::Terminals() : 1;
  _gpus = _nodes / _terms_per_gpu;

  string const order = config.GetStr( "trace_order" );
  if(order == "rr") {
    _round_robin = true;
  } else if(order == "fifo") {
    _round_robin = false;
  } else {
    Error( "collective: trace_order must be rr or fifo" );
  }

  _max_packet_size = _GetNextPacketSize(0);
  _header_flits = config.GetInt( "packet_header_flits" );
  if(_header_flits < 0) {
    Error( "collective: packet_header_flits must be >= 0" );
  }
  _max_cycles = config.GetInt( "trace_max_cycles" );
  _trace_out = config.GetStr( "trace_out" );
  _trace_packets_out = config.GetStr( "trace_packets_out" );
  _timeline_out = config.GetStr( "link_timeline_out" );
  _timeline_window = max(1, config.GetInt( "link_timeline_window" ));
  if(!_trace_packets_out.empty()) {
    _packets_out = new ofstream(_trace_packets_out.c_str());
    if(!*_packets_out) {
      Error( "collective: cannot write trace_packets_out " + _trace_packets_out );
    }
    *_packets_out << "pid,msg,src_term,dst_term,hops,size,ctime,itime,atime" << endl;
  }

  string const trace_file = config.GetStr( "trace_file" );
  if(trace_file.empty()) {
    Error( "collective: trace_file must be set" );
  }
  _LoadTrace( trace_file );

  _pending.resize(_nodes);
  _active.resize(_nodes);
  _next_msg.resize(_nodes, -1);
}

CollectiveTrafficManager::~CollectiveTrafficManager( )
{
  delete _packets_out;
}

void CollectiveTrafficManager::_LoadTrace( const string & filename )
{
  ifstream in(filename.c_str());
  if(!in) {
    Error( "collective: cannot open trace_file " + filename );
  }

  map<int, int> index;
  vector<vector<int> > dep_ids;
  string line;
  int lineno = 0;
  while(getline(in, line)) {
    ++lineno;
    size_t const first = line.find_first_not_of(" \t\r");
    if((first == string::npos) || (line[first] == '#')) {
      continue;
    }
    istringstream ls(line);
    Message m;
    int ndeps;
    if(!(ls >> m.id >> m.src >> m.dst >> m.size >> m.launch
         >> m.dim_order >> m.tie_mask >> ndeps)) {
      ostringstream err;
      err << "collective: malformed line " << lineno << " in " << filename;
      Error( err.str() );
    }
    vector<int> deps(ndeps);
    for(int i = 0; i < ndeps; ++i) {
      if(!(ls >> deps[i])) {
        ostringstream err;
        err << "collective: missing dependency on line " << lineno;
        Error( err.str() );
      }
    }
    if((m.src < 0) || (m.src >= _gpus) || (m.dst < 0) || (m.dst >= _gpus) ||
       ((m.src == m.dst) && (m.size > 0)) || (m.size < 0) || (m.launch < 0)) {
      ostringstream err;
      err << "collective: invalid message on line " << lineno
          << " (GPU ids must be in [0," << _gpus - 1 << "], src != dst, size >= 0)";
      Error( err.str() );
    }
    if(index.count(m.id)) {
      ostringstream err;
      err << "collective: duplicate message id " << m.id;
      Error( err.str() );
    }

    int src_port = 0;
    int dst_port = 0;
    if(_hyperx && (m.size > 0)) {
      src_port = hyperx::FirstHopPort( m.src, m.dst, m.dim_order, m.tie_mask );
      dst_port = hyperx::ArrivalPort( m.src, m.dst, m.dim_order, m.tie_mask );
    } else if(_torus && (m.size > 0)) {
      src_port = multilinktorus::FirstHopPort( m.src, m.dst, m.dim_order, m.tie_mask );
      dst_port = multilinktorus::ArrivalPort( m.src, m.dst, m.dim_order, m.tie_mask );
    }
    m.src_term = m.src * _terms_per_gpu + (src_port % _terms_per_gpu);
    m.dst_term = m.dst * _terms_per_gpu + (dst_port % _terms_per_gpu);
    m.pkts_total = (m.size + _max_packet_size - 1) / _max_packet_size;

    index[m.id] = _msgs.size();
    _msgs.push_back(m);
    dep_ids.push_back(deps);
  }

  for(size_t i = 0; i < _msgs.size(); ++i) {
    for(size_t j = 0; j < dep_ids[i].size(); ++j) {
      map<int, int>::const_iterator iter = index.find(dep_ids[i][j]);
      if(iter == index.end()) {
        ostringstream err;
        err << "collective: message " << _msgs[i].id << " depends on unknown message "
            << dep_ids[i][j];
        Error( err.str() );
      }
      _msgs[i].deps.push_back(iter->second);
      _msgs[iter->second].succs.push_back(i);
    }
  }

  long long flits = 0;
  long long pkts = 0;
  for(size_t i = 0; i < _msgs.size(); ++i) {
    flits += _msgs[i].size;
    pkts += _msgs[i].pkts_total;
  }
  cout << "Loaded " << _msgs.size() << " messages (" << flits << " payload flits + "
       << pkts * _header_flits << " header flits in " << pkts << " packets) from "
       << filename << " for " << _gpus << " GPUs x " << _terms_per_gpu
       << " terminals." << endl;
}

void CollectiveTrafficManager::_ResetState( )
{
  for(int n = 0; n < _nodes; ++n) {
    _pending[n] = tPendingQueue();
    _active[n].clear();
    _next_msg[n] = -1;
  }
  _released = 0;
  _done = 0;
  _completion_time = -1;
  for(size_t i = 0; i < _msgs.size(); ++i) {
    Message & m = _msgs[i];
    m.deps_left = m.deps.size();
    m.eligible = -1;
    m.first_inject = -1;
    m.finish = -1;
    m.pkts_sent = 0;
    m.pkts_done = 0;
    m.flits_sent = 0;
  }
  for(size_t i = 0; i < _msgs.size(); ++i) {
    if(_msgs[i].deps.empty()) {
      _Release(i, _start_time);
    }
  }
}

// Zero-size messages are barriers: they complete as soon as they are
// eligible, without injecting anything.
void CollectiveTrafficManager::_Release( int i, int time )
{
  Message & m = _msgs[i];
  m.eligible = max(_start_time + m.launch, time);
  ++_released;
  if(m.size == 0) {
    m.first_inject = m.eligible;
    _Complete(i, m.eligible);
  } else if(_stripe) {
    for(int t = 0; t < _terms_per_gpu; ++t) {
      _pending[m.src_term + t].push(make_pair(m.eligible, i));
    }
  } else {
    _pending[m.src_term].push(make_pair(m.eligible, i));
  }
}

void CollectiveTrafficManager::_Complete( int i, int time )
{
  Message & m = _msgs[i];
  m.finish = time;
  ++_done;
  for(size_t s = 0; s < m.succs.size(); ++s) {
    int const succ = m.succs[s];
    if(--_msgs[succ].deps_left == 0) {
      _Release(succ, time);
    }
  }
}

int CollectiveTrafficManager::_IssuePacket( int source, int cl )
{
  int const t = _qtime[source][cl];
  tPendingQueue & pending = _pending[source];
  while(!pending.empty() && (pending.top().first <= t)) {
    _active[source].push_back(pending.top().second);
    pending.pop();
  }
  // Striped messages sit on every terminal of the GPU; drop the ones whose
  // packets were all taken by the other terminals.
  while(!_active[source].empty() &&
        (_msgs[_active[source].front()].pkts_sent == _msgs[_active[source].front()].pkts_total)) {
    _active[source].pop_front();
  }
  if(_active[source].empty()) {
    return 0;
  }
  _next_msg[source] = _active[source].front();
  _packet_seq_no[source]++;
  return 1;
}

void CollectiveTrafficManager::_GeneratePacket( int source, int stype,
                                                int cl, int time )
{
  int const i = _next_msg[source];
  assert((i >= 0) && (_active[source].front() == i));
  Message & m = _msgs[i];
  _active[source].pop_front();

  int const payload = min(_max_packet_size, m.size - m.flits_sent);
  m.flits_sent += payload;
  int const size = payload + _header_flits;
  if(++m.pkts_sent < m.pkts_total) {
    if(_round_robin) {
      _active[source].push_back(i);
    } else {
      _active[source].push_front(i);
    }
  }

  int const pid = _cur_pid++;
  assert(_cur_pid);
  bool const watch = gWatchOut && (_packets_to_watch.count(pid) > 0);
  bool record = false;
  if((_sim_state == running) ||
     ((_sim_state == draining) && (time < _drain_time))) {
    record = _measure_stats[cl];
  }

  if(watch) {
    *gWatchOut << GetSimTime() << " | " << "node" << source << " | "
               << "Enqueuing packet " << pid << " of message " << m.id
               << " at time " << time << "." << endl;
  }

  for(int j = 0; j < size; ++j) {
    Flit * f = Flit::New();
    f->id = _cur_id++;
    assert(_cur_id);
    f->pid = pid;
    f->watch = watch | (gWatchOut && (_flits_to_watch.count(f->id) > 0));
    f->subnetwork = 0;
    f->src = source;
    f->ctime = time;
    f->record = record;
    f->cl = cl;
    f->type = Flit::ANY_TYPE;
    f->msg_id = i;
    f->dim_order = m.dim_order;
    f->tie_dir = m.tie_mask;

    _total_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    if(record) {
      _measured_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    }

    f->head = (j == 0);
    f->tail = (j == size - 1);
    f->dest = f->head ? (_stripe ? m.dst_term + (source - m.src_term) : m.dst_term) : -1;

    switch(_pri_type) {
    case class_based:
      f->pri = _class_priority[cl];
      break;
    case age_based:
      f->pri = numeric_limits<int>::max() - time;
      break;
    case sequence_based:
      f->pri = numeric_limits<int>::max() - _packet_seq_no[source];
      break;
    default:
      f->pri = 0;
    }
    assert(f->pri >= 0);
    f->vc = -1;

    _partial_packets[source][cl].push_back(f);
  }
}

void CollectiveTrafficManager::_RetireFlit( Flit *f, int dest )
{
  bool const head = f->head;
  bool const tail = f->tail;
  int const i = f->msg_id;
  int const itime = f->itime;

  if(_packets_out) {
    if(head) {
      PacketHead h;
      h.id = f->id;
      h.ctime = f->ctime;
      h.itime = itime;
      _packet_heads[f->pid] = h;
    }
    if(tail) {
      map<int, PacketHead>::iterator iter = _packet_heads.find(f->pid);
      assert(iter != _packet_heads.end());
      PacketHead const & h = iter->second;
      *_packets_out << f->pid << ',' << _msgs[i].id << ',' << f->src << ','
                    << dest << ',' << f->hops << ',' << f->id - h.id + 1 << ','
                    << h.ctime - _start_time << ',' << h.itime - _start_time << ','
                    << f->atime - _start_time << '\n';
      _packet_heads.erase(iter);
    }
  }

  TrafficManager::_RetireFlit(f, dest);

  assert((i >= 0) && (i < (int)_msgs.size()));
  Message & m = _msgs[i];
  if(head && ((m.first_inject < 0) || (itime < m.first_inject))) {
    m.first_inject = itime;
  }
  if(tail && (++m.pkts_done == m.pkts_total)) {
    _Complete(i, _time);
  }
}

vector<long long> CollectiveTrafficManager::_ChannelFlits( ) const
{
  return _Flits(_net[0]->GetChannels());
}

vector<long long> CollectiveTrafficManager::_Flits( const vector<FlitChannel *> & chans )
{
  vector<long long> flits(chans.size(), 0);
  for(size_t c = 0; c < chans.size(); ++c) {
    vector<int> const & act = chans[c]->GetActivity();
    for(size_t k = 0; k < act.size(); ++k) {
      flits[c] += act[k];
    }
  }
  return flits;
}

bool CollectiveTrafficManager::_SingleSim( )
{
  _start_time = _time;
  _sim_state = running;
  _ResetState( );
  vector<long long> const chan_before = _ChannelFlits();
  vector<long long> const inject_before = _Flits(_net[0]->GetInject());
  vector<long long> const eject_before = _Flits(_net[0]->GetEject());

  cout << "Running collective (" << _msgs.size() << " messages)..." << endl;

  ofstream * timeline = 0;
  vector<vector<FlitChannel *> > tl_chans;
  vector<vector<long long> > tl_prev;
  char const * const tl_kind[] = { "link", "inject", "eject" };
  if(!_timeline_out.empty()) {
    timeline = new ofstream(_timeline_out.c_str());
    if(!*timeline) {
      Error( "collective: cannot write link_timeline_out " + _timeline_out );
    }
    *timeline << "t,kind,index,flits" << endl;
    tl_chans.push_back(_net[0]->GetChannels());
    tl_chans.push_back(_net[0]->GetInject());
    tl_chans.push_back(_net[0]->GetEject());
    for(size_t k = 0; k < tl_chans.size(); ++k) {
      tl_prev.push_back(_Flits(tl_chans[k]));
    }
  }

  while(_done < (int)_msgs.size()) {
    _Step( );

    int const elapsed = _time - _start_time;
    if(timeline && ((elapsed % _timeline_window == 0) || (_done == (int)_msgs.size()))) {
      int const t0 = ((elapsed - 1) / _timeline_window) * _timeline_window;
      for(size_t k = 0; k < tl_chans.size(); ++k) {
        vector<long long> const now = _Flits(tl_chans[k]);
        for(size_t c = 0; c < now.size(); ++c) {
          *timeline << t0 << ',' << tl_kind[k] << ',' << c << ','
                    << now[c] - tl_prev[k][c] << '\n';
        }
        tl_prev[k] = now;
      }
    }

    if((_max_cycles > 0) && (_time - _start_time >= _max_cycles)) {
      cout << "trace_max_cycles reached with " << _msgs.size() - _done
           << " messages unfinished." << endl;
      break;
    }

    if((_released == _done) && (_done < (int)_msgs.size())) {
      bool idle = true;
      for(int c = 0; c < _classes && idle; ++c) {
        idle = _total_in_flight_flits[c].empty();
      }
      if(idle) {
        cout << "Schedule stuck: " << _msgs.size() - _done
             << " messages wait on dependencies that never finish." << endl;
        break;
      }
    }
  }

  int last = _start_time;
  for(size_t i = 0; i < _msgs.size(); ++i) {
    last = max(last, _msgs[i].finish);
  }
  _completion_time = last - _start_time + 1;
  delete timeline;

  cout << "Collective completed " << _done << "/" << _msgs.size()
       << " messages in " << _completion_time << " cycles." << endl;
  _ReportLinkLoad( chan_before, inject_before, eject_before );
  _WriteTraceOut( );

  _sim_state = draining;
  _drain_time = _time;

  UpdateStats();
  DisplayStats();
  return (_done == (int)_msgs.size());
}

void CollectiveTrafficManager::_ReportLinkLoad( const vector<long long> & before,
                                                const vector<long long> & inject_before,
                                                const vector<long long> & eject_before ) const
{
  vector<long long> const after = _ChannelFlits();
  long long max_flits = 0;
  long long total = 0;
  int used = 0;
  for(size_t c = 0; c < after.size(); ++c) {
    long long const f = after[c] - before[c];
    max_flits = max(max_flits, f);
    total += f;
    used += (f > 0);
  }
  if(!after.empty()) {
    cout << "Network links: busiest " << max_flits << " flits, mean "
         << (double)total / after.size() << " flits over " << after.size()
         << " links (" << used << " used)." << endl;
    if(max_flits > 0) {
      cout << "Completion / busiest-link flits = "
           << (double)_completion_time / max_flits << endl;
    }
  }

  vector<long long> const inject = _Flits(_net[0]->GetInject());
  vector<long long> const eject = _Flits(_net[0]->GetEject());
  long long max_inject = 0;
  long long max_eject = 0;
  for(size_t c = 0; c < inject.size(); ++c) {
    max_inject = max(max_inject, inject[c] - inject_before[c]);
  }
  for(size_t c = 0; c < eject.size(); ++c) {
    max_eject = max(max_eject, eject[c] - eject_before[c]);
  }
  long long const max_all = max(max_flits, max(max_inject, max_eject));
  cout << "Terminal channels: busiest inject " << max_inject << " flits, busiest eject "
       << max_eject << " flits over " << inject.size() << " terminals." << endl;
  if(max_all > 0) {
    cout << "Completion / busiest-channel flits (incl. terminals) = "
         << (double)_completion_time / max_all << endl;
  }
}

void CollectiveTrafficManager::_WriteTraceOut( ) const
{
  if(_trace_out.empty()) {
    return;
  }
  ofstream out(_trace_out.c_str());
  if(!out) {
    cerr << "collective: cannot write trace_out " << _trace_out << endl;
    return;
  }
  out << "id,src,dst,size_flits,launch,eligible,first_inject,finish,src_term,dst_term" << endl;
  for(size_t i = 0; i < _msgs.size(); ++i) {
    Message const & m = _msgs[i];
    out << m.id << ',' << m.src << ',' << m.dst << ',' << m.size << ','
        << m.launch << ',' << m.eligible - _start_time << ','
        << m.first_inject - _start_time << ',' << m.finish - _start_time << ','
        << m.src_term << ',' << m.dst_term << endl;
  }
}

void CollectiveTrafficManager::DisplayStats( ostream & os ) const
{
  TrafficManager::DisplayStats(os);
  os << "Collective completion time = " << _completion_time << " cycles" << endl;
}
