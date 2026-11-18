#pragma once
#include "source_state.hpp"
#include "large_queue.hpp"
#include "callback_counters.hpp"
#include "callback_diagnostics.hpp"
#include <typeinfo>
#include <vector>
#include <atomic>
#include <bit>
#include <cstring>
#include <exception>
#include <thread>
#include <span>
#include <memory>
#include <sys/resource.h>
namespace shadow::stage1 {
inline constexpr size_t TraceWireBytes=96;
inline constexpr bool MODEL_DECISIONS_ENABLED=false,PAPER_MUTATION_ENABLED=false,REAL_BROKER_ENABLED=false;
enum class State:uint8_t {UNINITIALIZED,CONFIGURED,STARTING,CATCHING_UP,FRESH_REALTIME,STOP_REQUESTED,STOPPING,STOPPED,FAILED};
enum class Cause:uint8_t {NONE,AUTHORIZED_CONTROLLED_STOP,SOURCE_FAILURE,APPLICATION_FAILURE,CALLBACK_FAILURE,SAFETY_FAILURE,FRESHNESS_LOST_AFTER_ACQUISITION,DISK_FLOOR_VIOLATION,MONITORING_EVIDENCE_FAILURE};
enum class Kind:uint16_t {SOURCE_CALLBACK=1,SOURCE_COORDINATE,FRESHNESS_TRANSITION,CANONICAL_CLOSE,STATE_PUBLICATION,FIRST_EXCEPTION,STOP_REQUEST,PROCESS_TAPE_RETURN,FINAL_SOURCE_STATE,NORMALIZED_EVENT_META,NORMALIZED_EVENT_VALUES,OBSERVATION_COMPLETED,LIFECYCLE_MILESTONE,MODEL_INPUT_READY=32,PUREJOB_ENQUEUE,GPU_DISPATCH,GPU_RESULT,COMMAND_ADMISSION,TOKEN_VALIDATION,DECISION,PAPER_INTENT,PAPER_FILL};
struct TraceRecord {uint16_t version=1;Kind kind{};uint32_t symbol=UINT32_MAX;uint64_t ordinal=0,sequence=0,finalization=0,session=0,epoch=0,generation=0;int64_t wall=0,feed=0,event=0,value=0;};
// Bounded MPSC trace. Producers make at most 32 reservation attempts and
// never wait for the disk consumer or another producer to publish its slot.
// One consumer serializes records in reservation order.
inline int64_t monotonic_now_ns();
template<size_t N> class Trace {
 struct Slot {std::atomic<uint64_t> turn{0};TraceRecord row{};int64_t emitted_ns=0;};
 std::unique_ptr<Slot[]> rows;alignas(64) std::atomic<uint64_t> write{0};alignas(64) uint64_t read=0;
public:
 std::atomic<uint64_t> drops{0},incomplete{0},first_drop_ordinal{0};
 // Producer and consumer progress must not invalidate the same cache line
 // on every record. Occupancy is intentionally shared and remains exact.
 alignas(64) std::atomic<uint64_t> records{0},high_water{0},first_emitted_ns{0};
 alignas(64) std::atomic<uint64_t> consumed{0},max_latency_ns{0};
 alignas(64) std::atomic<uint64_t> depth{0};
 static constexpr size_t capacity=N;
 const bool enabled;
 explicit Trace(bool active=true):rows(active?std::make_unique<Slot[]>(N):nullptr),enabled(active){if(active)for(size_t i=0;i<N;++i)rows[i].turn=i;}
 void emit(TraceRecord row)noexcept {
  if(!enabled)return;
  auto pos=write.load(std::memory_order_relaxed);
  for(int attempt=0;attempt<32;++attempt){auto& slot=rows[pos%N];auto turn=slot.turn.load(std::memory_order_acquire);
   if(turn==pos){if(write.compare_exchange_weak(pos,pos+1,std::memory_order_relaxed)){auto occupied=depth.fetch_add(1)+1;auto high=high_water.load();for(int update=0;high<occupied&&update<32;++update)high_water.compare_exchange_weak(high,occupied);if(high<occupied&&high_water.load()<occupied)++drops;row.ordinal=pos+1;slot.row=row;slot.emitted_ns=monotonic_now_ns();if(pos==0)first_emitted_ns=uint64_t(slot.emitted_ns);slot.turn.store(pos+1,std::memory_order_release);++records;return;}}
   else if(turn<pos){uint64_t zero=0;first_drop_ordinal.compare_exchange_strong(zero,pos+1);++drops;return;}else pos=write.load(std::memory_order_relaxed);
  }uint64_t zero=0;first_drop_ordinal.compare_exchange_strong(zero,pos+1);++drops;
 }
 bool pop(TraceRecord& row)noexcept {if(!enabled)return false;auto& slot=rows[read%N];if(slot.turn.load(std::memory_order_acquire)!=read+1)return false;row=slot.row;auto age=uint64_t(monotonic_now_ns()-slot.emitted_ns);if(age>max_latency_ns.load())max_latency_ns=age;depth.fetch_sub(1);slot.turn.store(read+N,std::memory_order_release);++read;++consumed;return true;}
 uint64_t pending()const noexcept{auto produced=records.load(),used=consumed.load();return produced>used?produced-used:0;}
 // Sink is called ONLY by the trace consumer; count short/failed records.
 template<class Sink>bool write_one(Sink sink){TraceRecord row;if(!pop(row))return false;try{if(sink(row)!=TraceWireBytes)++incomplete;}catch(...){++incomplete;}return true;}
};
template<class T,size_t N>class Spsc {
 static_assert(N>0);std::unique_ptr<T[]> rows=std::make_unique<T[]>(N);alignas(64)std::atomic<uint64_t> in{0};alignas(64)std::atomic<uint64_t> out{0};
 // Occupancy linearizes at accepted enqueue/removal, including a reserved
 // producer slot before publication. Only the producer updates the maximum.
 std::atomic<uint64_t> occupancy{0},maximum{0},overflows{0};
public:
 static constexpr size_t capacity=N;
 bool push(const T& row)noexcept{auto i=in.load(std::memory_order_relaxed);if(i-out.load(std::memory_order_acquire)==N){++overflows;return false;}rows[i%N]=row;auto occupied=occupancy.fetch_add(1)+1;if(occupied>maximum.load())maximum.store(occupied);in.store(i+1,std::memory_order_release);return true;}
 template<class Accept>bool pop_if(T& row,Accept accept)noexcept{auto o=out.load(std::memory_order_relaxed);if(o==in.load(std::memory_order_acquire)||!accept(rows[o%N]))return false;row=rows[o%N];occupancy.fetch_sub(1);out.store(o+1,std::memory_order_release);return true;}
 bool pop(T& row)noexcept{return pop_if(row,[](const T&){return true;});}
 // Consumer-only, synchronous borrowed view. Producer cannot reuse this slot
 // until fn returns. The reference must never escape fn; release on exceptions.
 template<class F>bool consume(F&& fn){auto o=out.load(std::memory_order_relaxed);if(o==in.load(std::memory_order_acquire))return false;
  auto release=[&]{occupancy.fetch_sub(1);out.store(o+1,std::memory_order_release);};
  try{fn(static_cast<const T&>(rows[o%N]));}catch(...){release();throw;}release();return true;
 }
 uint64_t accepted_count()const noexcept{return in.load();}
 uint64_t dequeued_count()const noexcept{return out.load();}
 uint64_t depth()const noexcept{return occupancy.load();}
 uint64_t high_water()const noexcept{return maximum.load();}
 uint64_t overflow_count()const noexcept{return overflows.load();}
};
struct ReferenceField {std::array<char,48> name{};uint8_t set=0,type=0;bool name_truncated=false,value_truncated=false;std::array<std::byte,128> value{};};
// Owns every byte; no SDK pointers escape callback. IDs remain integers.
struct Event {
 SdkCallbackMetadata sdk{};bool control_only=false;
 std::exception_ptr decoder_error;int64_t eligible_monotonic_ns=0;int utc_offset_seconds=0;SourceStamp stamp{};int64_t event_ns=0,wall_ns=0;uint32_t symbol=UINT32_MAX,message_type=0,exchange_sequence=0,records_back=0;
 uint32_t processing_date=0,session_date=0;uint16_t listed=0,reporting=0,symbol_atom=0,category=0;
 uint8_t session_id=0;CallbackKind callback_kind=CallbackKind::UNKNOWN;int status=0,mode=0;bool event_valid=false,accepted=false;
 std::array<char,64> name{};int32_t raw_price=0;uint64_t size=0,total_volume=0,fractional_size=0,fractional_total=0;
 uint8_t price_type=0,price_flags=0,condition_flags=0,trade_condition=0,volume_type=0,match_type=0;int32_t match_distance=0;
 double price=0;char side='N';uint16_t reference_count=0;std::array<ReferenceField,64> references{};
};
using Callback=int(*)(void*,const Event&);
struct Sdk {virtual ~Sdk()=default;virtual int process(Callback,void*)=0;};
template<class Sink>void emit_normalized_event(Sink& trace,const Event& e){
 TraceRecord meta;meta.kind=Kind::NORMALIZED_EVENT_META;meta.symbol=e.symbol;meta.sequence=e.stamp.sequence;meta.session=e.stamp.session;meta.epoch=e.stamp.source_epoch;meta.generation=e.stamp.generation;meta.finalization=e.stamp.execution_epoch;
 meta.wall=e.wall_ns;meta.feed=e.stamp.feed;meta.event=e.event_ns;
 meta.value=std::bit_cast<int64_t>(uint64_t(uint32_t(e.status))|(uint64_t(uint32_t(e.mode))<<32));trace.emit(meta);
 auto values=meta;values.kind=Kind::NORMALIZED_EVENT_VALUES;values.wall=e.eligible_monotonic_ns;values.feed=std::bit_cast<int64_t>(e.price);values.event=std::bit_cast<int64_t>(e.size);
 values.value=std::bit_cast<int64_t>(uint64_t(e.accepted)|(uint64_t(e.event_valid)<<1)|(uint64_t(uint8_t(e.side))<<8)|(uint64_t(uint32_t(e.utc_offset_seconds))<<32));trace.emit(values);
}

// Owns SDK exclusively, including dlclose after ProcessTape returns and all
// callbacks finish. No callback registration exists outside this object.
struct FirstCause {
 std::mutex mutex;std::atomic<Cause> type{Cause::NONE};std::atomic<int64_t> monotonic_ns{0};std::exception_ptr exception;std::atomic<bool> exception_present{false};std::array<char,192> message{};CallbackFailureDiagnostic diagnostic{};
 bool report(Cause c,const char* msg,std::exception_ptr e={},const CallbackFailureDiagnostic* details=nullptr){std::lock_guard guard(mutex);if(type.load()!=Cause::NONE)return false;monotonic_ns=monotonic_now_ns();exception=e;exception_present=bool(e);if(details)diagnostic=*details;std::strncpy(message.data(),msg,message.size()-1);type.store(c,std::memory_order_release);return true;}
};
// Fixed storage only. No formatting, allocation or I/O on callbacks.
enum RecoveryReason:uint32_t {RecoveryInit=1,RecoveryMode=2,RecoveryNoTrade=4,
 RecoveryFeedFuture=8,RecoveryFeedStale=16,RecoveryTradeFuture=32,
 RecoveryTradeStale=64,RecoveryControl=128,RecoveryExpired=256,RecoveryStop=512};
struct RecoveryRecord {
 int64_t mono=0,wall=0,feed=0,trade=0,candidate=-1,valid_until=0;
 uint64_t sequence=0,depth=0;uint32_t reasons=0;int mode=-1,status=-1,context=0,callback_kind=0;uint32_t message_type=0;
 bool was_fresh=false,initialized=false,frontier=false,candidate_reset=false;
};
struct RecoveryDiagnostics {
 static constexpr size_t Capacity=16;
 uint64_t invalid_checks=0,resets=0,entries=0,starts=0,completions=0,retained=0,overwritten=0;
 uint64_t feed_leads=0,tolerated_feed_leads=0;int64_t max_feed_lead_ns=0;
 std::array<uint64_t,10> reason_counts{};
 RecoveryRecord entry{},last_invalid{};std::array<RecoveryRecord,Capacity> recent{};
 int64_t candidate=-1,progress=0;
};
static_assert(sizeof(RecoveryDiagnostics)<=4096,"recovery diagnostics fixed memory ceiling");
struct Health {
 RecoveryDiagnostics recovery_diagnostics{};
 CallbackCounts callback_types{};
 uint64_t queue_high_water=0,trace_records=0,canonical_admitted=0,canonical_dequeued=0,status_admitted=0,status_noop_skipped=0,control_callbacks=0;
 int64_t last_callback_monotonic_ns=0,last_source_ns=0,last_event_ns=0,sampled_wall_ns=0,valid_until_monotonic_ns=0;
 int64_t closure_lag_ns=0,max_closure_lag_ns=0,latest_close_time_ns=0;
 uint64_t trace_depth=0,trace_high_water=0,trace_capacity=0,trace_max_latency_ns=0;
 uint64_t latest_close_symbol=UINT32_MAX,latest_close_sequence=0;
 bool source_returned=false,stop_acknowledged=false;int source_return_code=0;uint64_t stop_boundary_sequence=0,stop_ack_callbacks=0;
 bool exception_present=false;
 State state{};Cause cause{},terminal{};bool fresh=false,stop=false;uint64_t callbacks=0,sequence=0,finalization=0,depth=0,closures=0,warm=0,generation=0,session=0,epoch=0,trace_drops=0,trace_incomplete=0,overflow=0;int64_t last_callback=0,closure_lag=0;double callback_rate=0;uint64_t peak_rss_kib=0;
 // No such owners exist in Stage 1; zero values are explicitly unavailable.
 bool initialization_ready=false,validated_header_seen=false,loaded_state_seen=false,recoverable_service=false,frontier_acquired=false,recovering=false;int sdk_mode=-1,last_sdk_status=-1;uint32_t source_gate_mask=0;int initialization_policy=0;
 bool scanner_available=false,jobs_available=false,commands_available=false;uint64_t prewatch=0,promoted=0,model_ready=0,purejobs=0,command_depth=0,stale_outputs=0;bool paper_mutation=false;};
// Nanex documents approximately 1 ms feed-clock precision, independently of the host clock.
inline constexpr int64_t FeedFutureToleranceNs=1'000'000;
inline bool feed_future(int64_t wall,int64_t feed){return feed>wall&&uint64_t(feed)-uint64_t(wall)>FeedFutureToleranceNs;}
inline bool feed_fresh(int64_t wall,int64_t feed){return !feed_future(wall,feed)&&(feed>=wall||wall-feed<=5*NS);}
inline bool fresh_time(int64_t wall,int64_t event,int64_t feed){return wall>=event&&wall-event<=5*NS&&feed_fresh(wall,feed);}
inline int64_t wall_now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
inline int64_t monotonic_now_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline constexpr int64_t ObservationDurationSec=1200;
enum class InitializationPolicy:uint8_t {LoadedStateRequired,ValidatedHeaderOrState};
enum SourceGate:uint32_t {MissingInitialization=1,NotCurrentBuffer=2,MissingTradeTime=4,FeedTimeInvalid=8,TradeTimeInvalid=16,StopRequested=32,SourceFailed=64,Recovering=128};
inline const char* source_gate_reason(uint32_t f,bool service=false){if(f&SourceFailed)return "SOURCE_FAILURE";if(f&StopRequested)return "STOP_REQUESTED";if(f&MissingInitialization)return "INITIALIZATION_NOT_READY";if(f&NotCurrentBuffer)return service?"SOURCE_BUFFER_NOT_ELIGIBLE":"NOT_CURRENT_BUFFER_MODE6";if(f&MissingTradeTime)return "NO_TRADE_TIMESTAMP";if(f&FeedTimeInvalid)return "FEED_STALE_OR_FUTURE";if(f&TradeTimeInvalid)return "TRADE_STALE_OR_FUTURE";if(f&Recovering)return "RECOVERY_STABILIZING";return "READY";}
struct ObservationConfiguration {
 bool enabled=false;
 // Zero explicitly means unbounded pre-fresh operation. Never bounds observation.
 int64_t pre_fresh_timeout_sec=0;
 int64_t (*clock)()=monotonic_now_ns;
 int64_t duration_sec=ObservationDurationSec;
 bool completion_required=true;
 InitializationPolicy initialization_policy=InitializationPolicy::LoadedStateRequired;
 bool recoverable_service=false;
 int64_t recovery_stability_ns=2*NS;
};
struct ObservationStatus {
 bool acquired=false,complete=false;
 int64_t fresh_acquired_monotonic_ns=0,deadline_monotonic_ns=0;
 int64_t duration_sec=ObservationDurationSec;
 int64_t fresh_wall_ns=0;uint64_t fresh_source_sequence=0,fresh_finalization_sequence=0;
};
template<size_t Q=1024,size_t T=16384>class Owner {
 InitializationJournal initialization;std::vector<std::string> diagnostic_secrets;CallbackCounters callback_metrics;std::unique_ptr<Sdk> sdk;std::thread worker;mutable std::recursive_mutex control;std::atomic<bool> invoked{false},requested{false},ack{false},returned{false},fresh{false};std::atomic<int64_t> last_event{0},last_feed{0},last_wall{0},first_wall{0};
 std::atomic<Cause> terminal{Cause::NONE};std::atomic<uint64_t> stop_sequence{0},stop_ack_callbacks{0};
 // Producer-only canonical-admission key. Never crosses a market/control boundary.
 bool last_canonical_was_status=false;int64_t last_canonical_feed=0;int last_canonical_mode=0;uint64_t last_canonical_generation=0;
 std::atomic<uint64_t> status_admitted{0},status_noop_skipped{0},control_callbacks{0};
 std::atomic<int> mode{-1},last_sdk_status{-1};std::atomic<bool> loaded{false},validated_header{false};uint64_t source_sequence=0;int result=0;
 std::atomic<State> state{State::UNINITIALIZED};std::atomic<uint64_t> callbacks{0},finalization{0},closures{0},warm{0},generation{0},overflow{0};std::atomic<int64_t> closure_lag{0},max_closure_lag{0},last_callback_mono{0},latest_close_time{0};
 std::atomic<uint64_t> latest_close_symbol{UINT32_MAX},latest_close_sequence{0};
 ObservationConfiguration observation_config;ObservationStatus observation_status;std::atomic<int64_t> started_monotonic_ns{-1},fresh_valid_until_monotonic_ns{0};
 std::atomic<bool> frontier_acquired{false},recovering{false};int64_t recovery_candidate_ns=-1;
 int64_t feed_progress_deadline_ns=0,trade_progress_deadline_ns=0,feed_progress_ns=0,trade_progress_ns=0;
 RecoveryDiagnostics recovery_diagnostics;int recovery_callback_kind=0;uint32_t recovery_message_type=0;
 uint32_t recovery_reasons(int64_t wall,int64_t mono)const {
  uint32_t r=0;auto trade=last_event.load(),feed=last_feed.load();
  if(!initialization_ready())r|=RecoveryInit;
  if(!mode_eligible(mode))r|=RecoveryMode;
  if(!trade)r|=RecoveryNoTrade;
  if(feed_future(wall,feed))r|=RecoveryFeedFuture;else if(wall>feed&&wall-feed>5*NS)r|=RecoveryFeedStale;
  if(wall<trade)r|=RecoveryTradeFuture;else if(wall-trade>5*NS)r|=RecoveryTradeStale;
  if(requested)r|=RecoveryStop;
  if(fresh_valid_until_monotonic_ns>0&&mono>fresh_valid_until_monotonic_ns)r|=RecoveryExpired;
  return r;
 }
 void recovery_invalid(uint32_t reasons,int64_t mono,int64_t wall,int context,bool clears_candidate=true) {
  if(!observation_config.recoverable_service)return;
  auto& d=recovery_diagnostics;++d.invalid_checks;
  for(size_t i=0;i<d.reason_counts.size();++i)if(reasons&(1u<<i))++d.reason_counts[i];
  RecoveryRecord row{mono,wall,last_feed.load(),last_event.load(),recovery_candidate_ns,
   fresh_valid_until_monotonic_ns,source_sequence,queue.depth(),reasons,mode,last_sdk_status,context,recovery_callback_kind,recovery_message_type,
   fresh.load(),initialization_ready(),frontier_acquired.load(),clears_candidate&&recovery_candidate_ns>=0};
  d.last_invalid=row;
  if(fresh){++d.entries;d.entry=row;}
  if(fresh||recovery_candidate_ns>=0){
   if(row.candidate_reset)++d.resets;
   d.recent[d.retained%RecoveryDiagnostics::Capacity]=row;++d.retained;
   if(d.retained>RecoveryDiagnostics::Capacity)++d.overwritten;
  }
 }
 bool initialization_ready()const{return loaded.load()||(observation_config.initialization_policy==InitializationPolicy::ValidatedHeaderOrState&&validated_header.load());}
 bool mode_eligible(int m)const{return observation_config.recoverable_service&&frontier_acquired?m>=2&&m<=6:m==6;}
 bool service_stable(bool valid,int64_t mono,int64_t wall,int context){
  if(!observation_config.recoverable_service)return valid;
  if(!valid){recovery_invalid(recovery_reasons(wall,mono),mono,wall,context);recovery_candidate_ns=-1;recovering=false;return false;}
  if(fresh){recovering=false;return true;}
  if(recovery_candidate_ns<0){recovery_candidate_ns=mono;++recovery_diagnostics.starts;}
  bool stable=mono-recovery_candidate_ns>=observation_config.recovery_stability_ns;recovering=!stable;
  if(stable){frontier_acquired=true;++recovery_diagnostics.completions;}
  return stable;
 }
 void check_observation_continuity(int64_t mono){
  // Detect an expired freshness interval even if the control thread was delayed
  // past the observation deadline or a later callback would conceal the gap.
  if(observation_config.enabled&&observation_status.acquired&&!requested&&!observation_status.complete&&mono>fresh_valid_until_monotonic_ns&&fresh_valid_until_monotonic_ns<observation_status.deadline_monotonic_ns)
   failure(Cause::FRESHNESS_LOST_AFTER_ACQUISITION,"FRESHNESS_LOST_AFTER_ACQUISITION");
 }
 void complete_observation(int64_t mono){
  if(observation_config.enabled&&observation_status.acquired&&!observation_status.complete&&!requested&&mono>=observation_status.deadline_monotonic_ns){
   observation_status.complete=true;Event complete;complete.stamp.sequence=source_sequence;complete.stamp.feed=last_feed;complete.wall_ns=wall_now_ns();complete.event_ns=mono;record(Kind::OBSERVATION_COMPLETED,complete,mono,finalization.load());request_stop();
  }
 }
 void transition(bool next,const Event& e){if(fresh.exchange(next)!=next){
  if(observation_config.recoverable_service&&!next)recovery_candidate_ns=-1;
  if(next){++generation;}record(Kind::FRESHNESS_TRANSITION,e,next,finalization.load());
  if(!observation_config.enabled||requested)return;
  auto mono=observation_config.clock();
  if(next&&!observation_status.acquired){
   observation_status.acquired=true;observation_status.fresh_acquired_monotonic_ns=mono;
   observation_status.deadline_monotonic_ns=mono+observation_config.duration_sec*NS;
   observation_status.fresh_wall_ns=e.wall_ns;observation_status.fresh_source_sequence=e.stamp.sequence;observation_status.fresh_finalization_sequence=finalization.load();
  }else if(!next&&observation_status.acquired&&!observation_status.complete){
   if(mono<observation_status.deadline_monotonic_ns)failure(Cause::FRESHNESS_LOST_AFTER_ACQUISITION,"FRESHNESS_LOST_AFTER_ACQUISITION");
   else complete_observation(mono);
  }
 }}
 void check_pre_fresh_timeout(int64_t mono){
  if(observation_config.enabled&&!requested&&!observation_status.acquired&&observation_config.pre_fresh_timeout_sec>0&&started_monotonic_ns>=0&&mono-started_monotonic_ns.load()>=observation_config.pre_fresh_timeout_sec*NS)
   failure(Cause::SOURCE_FAILURE,"PRE_FRESH_TIMEOUT");
 }
 // Called only after this callback's required input has been admitted (or
 // proved idempotent). Progress the same immutable deadline without relying
 // on the consumer winning the callback's control mutex during a burst.
 int callback_completion(int64_t mono){
  if(fresh)complete_observation(mono);
  if(requested){acknowledge_stop();return 1;}
  return 0;
 }
 static int trampoline(void* p,const Event& e)noexcept{return static_cast<Owner*>(p)->callback(e);}
public:
 const uint64_t session,epoch;std::conditional_t<Q==0,LargeSourceQueue<Event>,Spsc<Event,Q>> queue;Trace<T> trace;FirstCause first;
 Owner(std::unique_ptr<Sdk> s,uint64_t session_,uint64_t epoch_,ObservationConfiguration observation={},std::vector<std::string> secrets={},bool full_trace=true):diagnostic_secrets(std::move(secrets)),sdk(std::move(s)),observation_config(observation),session(session_),epoch(epoch_),trace(full_trace){require(bool(sdk),"SDK required");state=State::CONFIGURED;}
 ~Owner(){if(worker.joinable()){request_stop();worker.join();}}
 ObservationStatus observation()const {std::lock_guard guard(control);return observation_status;}
 void start(){require(!worker.joinable()&&!invoked,"source already started");worker=std::thread([this]{run();});}
 void join(){require(worker.joinable(),"source not started");worker.join();}
 void record(Kind k,const Event& e,int64_t value=0,uint64_t fin=0)noexcept{if(!trace.enabled)return;TraceRecord r;r.kind=k;r.symbol=e.symbol;r.sequence=e.stamp.sequence;r.finalization=fin;r.session=session;r.epoch=epoch;r.generation=(k==Kind::SOURCE_CALLBACK||k==Kind::SOURCE_COORDINATE||k==Kind::CANONICAL_CLOSE||k==Kind::STATE_PUBLICATION)?e.stamp.generation:generation.load();r.wall=e.wall_ns;r.feed=e.stamp.feed;r.event=e.event_ns;r.value=value;trace.emit(r);}
 void failure(Cause c,const char* msg,std::exception_ptr e={},const CallbackFailureDiagnostic* details=nullptr){if(first.report(c,msg,e,details)){if(terminal.load()!=Cause::NONE){terminal=c;state=State::FAILED;}Event x;record(Kind::FIRST_EXCEPTION,x,int(c));}request_stop(false);}
 void milestone(State next){Event e;e.stamp.sequence=source_sequence;e.stamp.feed=last_feed;e.wall_ns=wall_now_ns();e.event_ns=observation_config.clock();record(Kind::LIFECYCLE_MILESTONE,e,int(next),finalization.load());}
 void acknowledge_stop(){if(!ack.exchange(true))milestone(State::STOPPING);state=State::STOPPING;}
 void request_stop(bool authorized=true){std::lock_guard guard(control);if(!authorized&&first.type==Cause::NONE)failure(Cause::SAFETY_FAILURE,"unauthorized stop");if(!requested.exchange(true)){state=State::STOP_REQUESTED;milestone(State::STOP_REQUESTED);stop_sequence=source_sequence;Event e;e.stamp.sequence=source_sequence;e.stamp.feed=last_feed;e.wall_ns=last_wall;record(Kind::STOP_REQUEST,e,authorized);transition(false,e);}}
 void fail_callback(Cause cause,const char* msg,const Event& e,const char* operation,std::exception_ptr ep={},const std::exception* error=nullptr){
  CallbackFailureDiagnostic d;d.available=true;d.sdk=e.sdk;d.sdk_decoder=bool(e.decoder_error);d.sequence=e.stamp.sequence;d.session=session;d.epoch=epoch;d.generation=generation.load();d.lifecycle=int(state.load());d.monotonic_ns=monotonic_now_ns();d.wall_ns=wall_now_ns();d.last_source_feed_ns=last_feed;d.source_feed_ns=e.stamp.feed;d.source_coordinate_available=e.stamp.feed!=0;
  diagnostic_copy(d.operation,e.decoder_error&&e.sdk.operation?e.sdk.operation:operation);
  diagnostic_copy(d.category,error?(dynamic_cast<const std::logic_error*>(error)?"std::logic_error":dynamic_cast<const std::runtime_error*>(error)?"std::runtime_error":"std::exception"):(ep?"non_std_exception":"failure_without_exception"));
  if(error){const auto* text=error->what();d.message_available=text!=nullptr;if(text)d.truncated=diagnostic_copy(d.message,text,diagnostic_secrets);diagnostic_copy(d.type,typeid(*error).name());}
  else if(!ep){d.message_available=true;d.truncated=diagnostic_copy(d.message,msg,diagnostic_secrets);}
  failure(cause,msg,ep,&d);
 }
 const InitializationJournal& initialization_diagnostics()const{require(returned&&!worker.joinable(),"diagnostics require source join");return initialization;}
 int callback(Event e)noexcept {const char* operation="callback_entry";try{std::lock_guard guard(control);
  if(!callback_metrics.record(e.callback_kind)){fail_callback(Cause::MONITORING_EVIDENCE_FAILURE,"callback counter overflow",e,"callback_counter_accounting");acknowledge_stop();return 1;}
  e.stamp={session,epoch,generation.load(),0,++source_sequence,source_sequence,e.stamp.feed,uint8_t(e.mode)};
  last_sdk_status=e.status;recovery_callback_kind=int(e.callback_kind);recovery_message_type=e.message_type;last_callback_mono=observation_config.clock();++callbacks;last_wall=e.wall_ns;if(!first_wall.load())first_wall=e.wall_ns;record(Kind::SOURCE_CALLBACK,e);
  if(trace.enabled&&e.sdk.available&&!initialization.market_seen){
   if(e.callback_kind==CallbackKind::STATUS){if(initialization.count<initialization.entries.size())initialization.entries[initialization.count++]={e.sdk,e.stamp.sequence,last_callback_mono.load(),e.wall_ns};else initialization.truncated=true;}
   if(e.callback_kind==CallbackKind::TRADE||e.callback_kind==CallbackKind::QUOTE)initialization.market_seen=true;
  }
  if(e.decoder_error)std::rethrow_exception(e.decoder_error);
  if(e.status==4){fail_callback(Cause::SOURCE_FAILURE,"SDK error status",e,"SDK error status");acknowledge_stop();return 1;}
  if(requested){++stop_ack_callbacks;acknowledge_stop();return 1;}
  operation="source_reset_validation";require(e.status!=3&&e.status!=6,"source reset requires new owner");
  if(observation_config.recoverable_service&&frontier_acquired)require(e.status!=1&&e.status!=7&&e.status!=10,"source reinitialization after frontier requires new owner");
  if(!e.control_only){operation="feed_regression_validation";require(e.stamp.feed>=last_feed,"feed regression");}
  operation="header_readiness_validation";
  if(e.status==10){
   bool valid=e.control_only&&e.callback_kind==CallbackKind::STATUS&&e.sdk.available&&e.sdk.message_type==0&&e.sdk.date_available&&e.processing_date==session&&e.sdk.date==session;
   if(observation_config.initialization_policy==InitializationPolicy::ValidatedHeaderOrState)require(valid,"unvalidated TAPEOPEN initialization");
   if(valid)validated_header=true;
  }
  // A header is not a synthesized LOADED_STATE. Duplicate state-load checks
  // remain independent of header readiness and apply to real continuations.
  operation="loaded_state_validation";if(e.status==7){require(!loaded,"duplicate loaded state");loaded=true;}
  if(e.control_only){++control_callbacks;last_canonical_was_status=false;mode=-1;auto mono=observation_config.clock();check_observation_continuity(mono);check_pre_fresh_timeout(mono);recovery_invalid(RecoveryControl,mono,e.wall_ns,3);recovery_candidate_ns=-1;recovering=false;transition(false,e);if(requested){acknowledge_stop();return 1;}auto prior=state.load();while(prior!=State::STOP_REQUESTED&&prior!=State::STOPPING&&!state.compare_exchange_weak(prior,State::CATCHING_UP)){}return callback_completion(mono);}
  e.eligible_monotonic_ns=observation_config.clock();
  // A later fresh callback must not conceal an interval that already expired.
  if(observation_config.recoverable_service&&(fresh||recovery_candidate_ns>=0)&&e.eligible_monotonic_ns>fresh_valid_until_monotonic_ns){recovery_invalid(RecoveryExpired,e.eligible_monotonic_ns,e.wall_ns,4);transition(false,e);recovery_candidate_ns=-1;}

  if(e.event_valid){last_event=e.event_ns;}if(!e.event_valid){e.event_ns=last_event;}last_feed=e.stamp.feed;mode=e.status==0?e.mode:-1;
  auto mono=observation_config.clock();check_observation_continuity(mono);check_pre_fresh_timeout(mono);
  if(requested){acknowledge_stop();return 1;}
  if(observation_config.recoverable_service&&e.stamp.feed>e.wall_ns){
   auto lead=e.stamp.feed-e.wall_ns;auto& d=recovery_diagnostics;++d.feed_leads;
   if(lead<=FeedFutureToleranceNs)++d.tolerated_feed_leads;
   d.max_feed_lead_ns=std::max(d.max_feed_lead_ns,lead);
  }
  bool valid=initialization_ready()&&mode_eligible(mode)&&last_event.load()!=0&&fresh_time(e.wall_ns,last_event,e.stamp.feed);
  if(valid||(observation_config.recoverable_service&&last_event.load()!=0&&fresh_time(e.wall_ns,last_event,e.stamp.feed))){
   auto feed_deadline=mono+5*NS-std::max(int64_t(0),e.wall_ns-e.stamp.feed);
   auto trade_deadline=mono+5*NS-(e.wall_ns-last_event.load());
   if(observation_config.recoverable_service){
    // Only timestamp progress renews a deadline; duplicate callbacks and wall rewinds cannot hide a stall.
    if(!feed_progress_deadline_ns||e.stamp.feed>feed_progress_ns){feed_progress_ns=e.stamp.feed;feed_progress_deadline_ns=feed_deadline;}
    else feed_progress_deadline_ns=std::min(feed_progress_deadline_ns,feed_deadline);
    if(!trade_progress_deadline_ns||last_event.load()>trade_progress_ns){trade_progress_ns=last_event;trade_progress_deadline_ns=trade_deadline;}
    else trade_progress_deadline_ns=std::min(trade_progress_deadline_ns,trade_deadline);
    fresh_valid_until_monotonic_ns=std::min(feed_progress_deadline_ns,trade_progress_deadline_ns);
    valid=valid&&mono<=fresh_valid_until_monotonic_ns;
   }else fresh_valid_until_monotonic_ns=std::min(feed_deadline,trade_deadline);
  }
  bool eligible=service_stable(valid,mono,e.wall_ns,1);
  transition(eligible,e);
  if(requested){acknowledge_stop();return 1;}
  auto prior=state.load();while(prior!=State::STOP_REQUESTED&&prior!=State::STOPPING&&!state.compare_exchange_weak(prior,eligible?State::FRESH_REALTIME:State::CATCHING_UP)){}
  record(Kind::SOURCE_COORDINATE,e);
  if(trace.enabled)emit_normalized_event(trace,e);
  // Only consecutive, fully validated RUNNING watermarks with identical feed,
  // mode and generation are idempotent. Keep the full raw/normalized journal.
  // The first STATUS after ANY market input is retained: a late first trade can
  // make a bar eligible at the same feed time. Every feed advance is retained.
  const bool clock_status=e.sdk.available&&e.callback_kind==CallbackKind::STATUS&&e.status==0;
  if(clock_status&&last_canonical_was_status&&e.stamp.feed==last_canonical_feed&&e.mode==last_canonical_mode&&e.stamp.generation==last_canonical_generation){++status_noop_skipped;return callback_completion(mono);}
  if(!queue.push(e)){++overflow;fail_callback(Cause::SAFETY_FAILURE,"bounded source queue overflow",e,"source_queue_enqueue");acknowledge_stop();return 1;}
  last_canonical_was_status=clock_status;last_canonical_feed=e.stamp.feed;last_canonical_mode=e.mode;last_canonical_generation=e.stamp.generation;if(clock_status)++status_admitted;
  return callback_completion(mono);
 }catch(const std::exception& error){fail_callback(Cause::CALLBACK_FAILURE,"callback exception",e,operation,std::current_exception(),&error);acknowledge_stop();return 1;}
 catch(...){fail_callback(Cause::CALLBACK_FAILURE,"callback exception",e,operation,std::current_exception());acknowledge_stop();return 1;}}
 // Invoked on the dedicated source thread. Caller joins it before finish().
 void run(){if(invoked.exchange(true)){failure(Cause::SAFETY_FAILURE,"source already invoked");return;}State initial=State::CONFIGURED;state.compare_exchange_strong(initial,State::STARTING);started_monotonic_ns=observation_config.clock();try{result=sdk->process(trampoline,this);}catch(...){failure(Cause::SOURCE_FAILURE,"ProcessTape threw",std::current_exception());result=-3;}Event e;e.stamp.sequence=source_sequence;e.stamp.feed=last_feed;e.wall_ns=last_wall;record(Kind::PROCESS_TAPE_RETURN,e,result);returned=true;}
 bool process_returned()const noexcept{return returned.load();}
 bool stop_requested()const noexcept{return requested.load();}
 // A consumer must not wait behind the producer before draining its queue.
 // If busy, callbacks perform continuity/deadline checks under this same lock;
 // a quiet source leaves it available for expiry and pre-fresh timeout checks.
 bool try_poll_freshness(int64_t now){
  return try_poll_freshness_at([now]{return now;});
 }
 template<class WallClock> bool try_poll_freshness_at(WallClock wall_clock){
  // Only inactive catch-up can skip polling. A recovering candidate must finish
  // its dwell even without callbacks; fresh states must still expire promptly.
  if(!fresh&&!requested&&!(observation_config.recoverable_service&&recovering)){
   if(!observation_config.enabled||observation_config.pre_fresh_timeout_sec==0)return false;
   auto started=started_monotonic_ns.load();
   if(started<0||observation_config.clock()-started<observation_config.pre_fresh_timeout_sec*NS)return false;
  }
  std::unique_lock guard(control,std::try_to_lock);
  if(!guard.owns_lock())return false;
  // A callback may advance feed time while a caller is being scheduled.
  // Read the clock after acquiring the same lock used by that callback.
  poll_freshness_locked(wall_clock());
  return true;
 }
 void poll_freshness(int64_t now){std::lock_guard guard(control);poll_freshness_locked(now);}
private:
 void poll_freshness_locked(int64_t now){
  if(observation_config.recoverable_service){
   Event e;e.wall_ns=now;e.stamp.feed=last_feed;e.stamp.sequence=source_sequence;
   auto mono=observation_config.clock();bool valid=!requested&&initialization_ready()&&mode_eligible(mode)&&last_event.load()!=0&&mono<=fresh_valid_until_monotonic_ns&&fresh_time(now,last_event,last_feed);
   bool next=service_stable(valid,mono,now,2);transition(next,e);
   auto prior=state.load();while(prior!=State::STOP_REQUESTED&&prior!=State::STOPPING&&!state.compare_exchange_weak(prior,next?State::FRESH_REALTIME:State::CATCHING_UP)){}
   return;
  }
  check_observation_continuity(observation_config.clock());
  if(fresh&&!fresh_time(now,last_event,last_feed)){Event e;e.wall_ns=now;transition(false,e);State expected=State::FRESH_REALTIME;state.compare_exchange_strong(expected,State::CATCHING_UP);}
  if(!observation_config.enabled||requested)return;
  auto mono=observation_config.clock();
  check_pre_fresh_timeout(mono);
  if(fresh)complete_observation(mono);
 }
public:
 template<class WallClock> Health health_at(WallClock wall_clock)const {
  // The one-second service monitor can wait for a coherent source snapshot.
  // Queue consumption continues to use the nonblocking freshness poll.
  std::lock_guard guard(control);
  auto h=health(wall_clock());
  h.recovery_diagnostics=recovery_diagnostics;
  h.recovery_diagnostics.candidate=fresh?-1:recovery_candidate_ns;
  if(!fresh&&recovery_candidate_ns>=0)
   h.recovery_diagnostics.progress=std::max<int64_t>(0,observation_config.clock()-recovery_candidate_ns);
  return h;
 }
 Cause finish(){require(terminal==Cause::NONE,"source already finalized");require(returned&&!worker.joinable()&&queue.depth()==0,"join and state drain required");if(observation_config.enabled&&observation_config.completion_required&&!observation_status.complete&&first.type==Cause::NONE)failure(Cause::SAFETY_FAILURE,"continuous fresh observation incomplete");Cause c=first.type;if(c==Cause::NONE){if(requested&&ack&&(result==1||result==-3))c=Cause::AUTHORIZED_CONTROLLED_STOP;else{failure(Cause::SOURCE_FAILURE,"unacknowledged stop or unexpected source return");c=first.type;}}terminal=c;fresh=false;state=c==Cause::AUTHORIZED_CONTROLLED_STOP?State::STOPPED:State::FAILED;Event e;e.stamp.sequence=source_sequence;e.stamp.feed=last_feed;e.wall_ns=last_wall;record(Kind::FINAL_SOURCE_STATE,e,int(state.load()));milestone(state.load());return c;}
 void warm_symbol(){++warm;}
 void publication(const Event& e,const CanonicalBar& b,uint32_t symbol,uint64_t contributor){++closures;finalization=e.stamp.sequence;Event x=e;x.symbol=symbol;x.event_ns=b.ts;x.stamp.sequence=contributor;record(Kind::CANONICAL_CLOSE,x,0,e.stamp.sequence);record(Kind::STATE_PUBLICATION,x,0,e.stamp.sequence);
  // The first source event making this closure eligible carries its source
  // acceptance time (before updating last_feed/last_event). Covers watermark and later-trade closures, one by one.
  auto lag=observation_config.clock()-e.eligible_monotonic_ns;
  require(lag>=0,"closure monotonic clock regression");
  closure_lag=lag;if(lag>max_closure_lag.load())max_closure_lag=lag;
  latest_close_time=b.ts;latest_close_symbol=symbol;latest_close_sequence=e.stamp.sequence;
 }
 Health health(int64_t now)const {Health h;h.source_returned=returned;h.source_return_code=h.source_returned?result:0;h.stop_acknowledged=ack;h.stop_boundary_sequence=stop_sequence;h.stop_ack_callbacks=stop_ack_callbacks;h.initialization_ready=initialization_ready();h.validated_header_seen=validated_header;h.loaded_state_seen=loaded;h.initialization_policy=int(observation_config.initialization_policy);h.sdk_mode=mode;h.last_sdk_status=last_sdk_status;h.recoverable_service=observation_config.recoverable_service;h.frontier_acquired=frontier_acquired;h.recovering=recovering;
 h.sampled_wall_ns=now;h.valid_until_monotonic_ns=fresh_valid_until_monotonic_ns;auto gate_feed=last_feed.load(),gate_event=last_event.load();if(!h.initialization_ready)h.source_gate_mask|=MissingInitialization;if(!mode_eligible(h.sdk_mode))h.source_gate_mask|=NotCurrentBuffer;if(!gate_event)h.source_gate_mask|=MissingTradeTime;if(!feed_fresh(now,gate_feed))h.source_gate_mask|=FeedTimeInvalid;if(now<gate_event||now-gate_event>5*NS)h.source_gate_mask|=TradeTimeInvalid;if(requested)h.source_gate_mask|=StopRequested;if(first.type!=Cause::NONE)h.source_gate_mask|=SourceFailed;if(h.recovering)h.source_gate_mask|=Recovering;
 h.callback_types=callback_metrics.snapshot();h.state=state;h.cause=first.type;h.terminal=terminal;h.fresh=!requested&&fresh&&fresh_time(now,gate_event,gate_feed)&&(!observation_config.recoverable_service||observation_config.clock()<=fresh_valid_until_monotonic_ns);h.stop=requested;h.callbacks=callbacks;h.sequence=h.callbacks;h.finalization=finalization;h.depth=queue.depth();h.closures=closures;h.warm=warm;h.generation=generation;h.session=session;h.epoch=epoch;h.last_callback=last_wall;h.closure_lag=closure_lag;h.closure_lag_ns=closure_lag;h.max_closure_lag_ns=max_closure_lag;
 h.canonical_admitted=queue.accepted_count();h.canonical_dequeued=queue.dequeued_count();h.status_admitted=status_admitted;h.status_noop_skipped=status_noop_skipped;h.control_callbacks=control_callbacks;
 h.trace_depth=trace.depth;h.trace_high_water=trace.high_water;h.trace_capacity=T;h.trace_max_latency_ns=trace.max_latency_ns;h.queue_high_water=queue.high_water();h.trace_records=trace.records;h.last_callback_monotonic_ns=last_callback_mono;h.last_source_ns=gate_feed;h.last_event_ns=gate_event;
 h.latest_close_time_ns=latest_close_time;h.latest_close_symbol=latest_close_symbol;h.latest_close_sequence=latest_close_sequence;h.exception_present=first.exception_present;h.trace_drops=trace.drops;h.trace_incomplete=trace.incomplete;h.overflow=overflow;auto elapsed=now-first_wall.load();h.callback_rate=elapsed>0?double(h.callbacks)*NS/elapsed:0;struct rusage usage{};if(getrusage(RUSAGE_SELF,&usage)==0)h.peak_rss_kib=uint64_t(usage.ru_maxrss);return h;}
};
// State thread only. Diagnostic symbol IDs are predeclared in lexical order.
// Canonical T1 and N1 are the unchanged qualified owners. No scanner/model/
// command/PAPER/broker object or callable is accepted by this type.
template<size_t Symbols,class Source,bool FastUninitialized=true,class Before=std::less<size_t>>struct StateEngine {
 CanonicalCore<Symbols> canonical;ProvenanceCore<Symbols> provenance;std::unique_ptr<FeatureState[]> features=std::make_unique<FeatureState[]>(Symbols);Source& owner;bool has_contribution=false;uint64_t accepted_market_events=0;
 Before before;
 StateEngine(Source& s,Before order={}):canonical(s.session,s.epoch),provenance(s.session,s.epoch),owner(s),before(order){}
 template<class Observe>bool step(Observe observe){Event e;if(!owner.queue.pop(e))return false;process(e,observe);return true;}
 template<class Observe>bool step_in_place(Observe&& observe){return owner.queue.consume([&](const Event& e){process(e,observe);});}
 template<class Observe>void process(const Event& e,Observe& observe){try{
  ClosureKind kind=ClosureKind::Watermark;
  auto emit=[&](size_t sym,CanonicalBar b){
   require(provenance.symbols[sym].current==b.ts,"canonical/provenance closure mismatch");
   provenance.close(uint32_t(sym),e.stamp,kind,[&](const BarCoordinate& coordinate){
    if(features[sym].warm.size==0)owner.warm_symbol();
    features[sym].append(b,int((b.ts/NS+e.utc_offset_seconds)%86400));
    owner.publication(e,b,uint32_t(sym),coordinate.source_sequence);observe(sym,b,e);
   });
  };
  // Before the first accepted trade both cores have current=-1 for every
  // symbol. Only trade/contribution can initialize a bar; an empty watermark
  // cannot emit closures or change history. Keep the identical source checks
  // and every coordinate update, without rescanning960 empty symbol slots.
  if(FastUninitialized&&!has_contribution){
   canonical.advance_source(e.stamp.session,e.stamp.source_epoch,e.stamp.feed);
   provenance.advance_source(e.stamp);
  }else{
   canonical.tick(e.stamp.session,e.stamp.source_epoch,e.stamp.feed,emit,before);
   provenance.delivery(e.stamp,[](auto){require(false,"unmatched provenance watermark");},before);
  }
  if(e.accepted){++accepted_market_events;require(e.symbol<Symbols,"diagnostic symbol bound");kind=ClosureKind::Trade;has_contribution=true;
   canonical.symbols[e.symbol].trade(e.event_ns,e.price,double(e.size),e.side,[&](auto b){emit(e.symbol,b);});
   provenance.contribution(e.symbol,e.event_ns,e.stamp,[](auto){require(false,"unmatched provenance trade");});
  }
 }catch(...){owner.failure(Cause::APPLICATION_FAILURE,"state engine failure",std::current_exception());}}
};
struct Simulator final:Sdk {
 std::span<const Event> corpus;int normal_result=-4,stop_result=-3;void(*after)(void*,size_t)=nullptr;void* context=nullptr;
 explicit Simulator(std::span<const Event> c):corpus(c){}
 int process(Callback cb,void* p)override{for(size_t i=0;i<corpus.size();++i){if(cb(p,corpus[i]))return stop_result;if(after)after(context,i);}return normal_result;}
};
}
