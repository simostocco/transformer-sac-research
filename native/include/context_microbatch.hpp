#pragma once
#include "slot_limits.hpp"

#include "eligibility.hpp"
#include "portfolio_capture.hpp"

#include <bitset>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>

namespace shadow::minimal {

// A source observation owns one immutable
// history. The worker owns the caches and the only GPU socket. No symbol lease
// or per-symbol model copy is needed to reuse a finalized context.
class ContextMicrobatch {
 using Job = integrated::Job;
 using Outcome = integrated::Outcome;
 using Snapshot = std::shared_ptr<const Job>;
 using TF = NativeFlatObservation::TF;
 static constexpr size_t Batch=20, History=65, RingSize=128, LaneDepth=4;
 static constexpr int64_t TransformerBudget=320'000'000, SacBudget=60'000'000;
 static constexpr int64_t PostTransformerReserve=40'000'000;

 struct Cache {
  int64_t bar=-1;
  uint64_t session=0, epoch=0, generation=0, execution=0;
  pure::v2::PreparedInput input{};
  TF value{};
 };
 struct Candidate {
  Snapshot job;
  std::bitset<History> missing;
  int64_t deadline=0;
 };
 enum Phase { Cold, Refill, Steady, Phases };
 struct Context { Snapshot job; size_t index; Phase phase; };
 struct PhaseStats {uint64_t rows=0,exclusive_calls=0;double cuda_ms=0,exclusive_ms=0;};
 struct Decision {
  Snapshot job;
  Outcome outcome;
  std::array<float,490> observation{};
 };

 std::mutex mutex_;
 std::array<std::deque<Snapshot>,stage1::SmallCapSlots> pending_{};
 std::array<int64_t,stage1::SmallCapSlots> last_published_{},last_admitted_{};
 std::array<uint64_t,stage1::SmallCapSlots> served_{};
 std::array<std::atomic<uint64_t>,stage1::SmallCapSlots> primed_{};
 std::array<SymbolGateTelemetry,stage1::SmallCapSlots>& gates_;
 std::array<std::atomic<uint64_t>,stage1::SmallCapSlots>& deferred_,& warm_deferred_;
 std::atomic<uint64_t>& total_deferred_;
 std::atomic<uint64_t>& total_warm_deferred_;
 PortfolioCapture* capture_;
 std::array<PhaseStats,Phases> phases_{};
 uint64_t tf_rows_=0,tf_calls_=0,mixed_calls_=0,expired_pending_=0;
 double tf_cuda_ms_=0,mixed_cuda_ms_=0;
 std::array<uint64_t,Batch+1> batch_sizes_{};
 std::array<int64_t,8> recent_tf_roundtrip_ns_{};
 int64_t max_tf_roundtrip_ns_=0;
 std::unique_ptr<std::array<Cache,stage1::SmallCapSlots*RingSize>> cache_=
  std::make_unique<std::array<Cache,stage1::SmallCapSlots*RingSize>>();
 uint64_t wire_=0, background_=0, lookup_ns_=0, prepare_ns_=0, verify_ns_=0, replaced_=0, timed_out_=0, at_stop_=0;
 NativeFlatObservation builder_;

 static int64_t end(const Job& j,size_t i){return j.id.bar-int64_t(History-1-i)*NS;}
 Cache& entry(const Job& j,size_t i){return (*cache_)[size_t(j.id.symbol)*RingSize+size_t(end(j,i)/NS)%RingSize];}
 bool cached(const Job& j,size_t i) {
  const auto& c=entry(j,i);const auto& id=j.id;
  return c.bar==end(j,i)&&c.session==id.session&&c.epoch==id.epoch&&c.generation==id.generation&&
         c.execution==id.execution_epoch;
 }
 std::bitset<History> misses(const Job& job){
  std::bitset<History> missing;
  for(size_t i=0;i<History;++i)missing[i]=!cached(job,i);
  return missing;
 }
 void defer(const Job& j){
  if(primed_[j.id.symbol]!=j.id.generation+1){++warm_deferred_[j.id.symbol];++total_warm_deferred_;}
  else{++deferred_[j.id.symbol];++total_deferred_;}
 }
 bool take(const Snapshot& job,Decision& d,int64_t now){
  const auto slot=job->id.symbol;
  std::lock_guard lock(mutex_);
  auto& lane=pending_[slot];
  if(lane.empty()||lane.front()!=job)return false;
  lane.pop_front();--depth;
  require(job->id.bar>last_admitted_[slot],"microbatch symbol admission regression");
  last_admitted_[slot]=job->id.bar;
  d.job=job;
  auto& r=d.outcome;
  r.id=job->id;r.id.job=++admitted;r.eligible=job->eligible;r.canonical=job->canonical;
  r.features=job->features;r.enqueued=job->enqueued;r.event_started=job->event_started;r.started=now;
  ++gates_[slot].accepted;++gates_[slot].model_started;
  return true;
 }

public:
 struct Clock {int64_t mono,wall;uint64_t generation;bool valid;};
 static constexpr size_t StorageBytes=stage1::SmallCapSlots*RingSize*sizeof(Cache)
                                     +((LaneDepth+1)*stage1::SmallCapSlots+1)*sizeof(Job);
 static constexpr size_t PendingLimit=LaneDepth*stage1::SmallCapSlots;
 std::atomic<uint64_t> admitted{0},offered{0},depth{0},high_water{0};
 uint64_t peak_batch=0;
 ContextMicrobatch(std::array<SymbolGateTelemetry,stage1::SmallCapSlots>& gates,
                   std::array<std::atomic<uint64_t>,stage1::SmallCapSlots>& deferred,
                   std::array<std::atomic<uint64_t>,stage1::SmallCapSlots>& warm_deferred,
                   std::atomic<uint64_t>& total_deferred,std::atomic<uint64_t>& total_warm_deferred,
                   PortfolioCapture* capture)
  :gates_(gates),deferred_(deferred),warm_deferred_(warm_deferred),total_deferred_(total_deferred),
   total_warm_deferred_(total_warm_deferred),capture_(capture){}

 bool primed(size_t slot,uint64_t generation) const {return primed_[slot]==generation+1;}

 void publish(const Job& job) {
  const auto slot=job.id.symbol;
  require(slot<pending_.size(),"microbatch symbol out of bounds");
  auto snapshot=std::make_shared<const Job>(job);
  std::lock_guard lock(mutex_);
  require(job.id.bar>last_published_[slot],"microbatch symbol publication regression");
  last_published_[slot]=job.id.bar;
  auto& lane=pending_[slot];
  if(lane.size()==LaneDepth){
   defer(*lane.front());++replaced_;
   timed_out_+=integrated::expired(lane.front()->eligible,job.enqueued);
   lane.pop_front();--depth;
  }
  lane.push_back(std::move(snapshot));++offered;
  const auto size=++depth;
  if(size>high_water)high_water=size;
 }

 void close() {
  std::lock_guard lock(mutex_);
  for(auto& lane:pending_)while(!lane.empty()){
   defer(*lane.front());lane.pop_front();--depth;++at_stop_;
  }
  require(depth==0&&offered==admitted+total_deferred_+total_warm_deferred_,"microbatch opportunity accounting");
 }

 std::string profile() const {
  std::ostringstream out;
  out<<std::setprecision(12)<<"{\"mode\":\"context_microbatch\",\"offers\":"<<offered<<",\"admitted\":"<<admitted
     <<",\"superseded\":"<<replaced_<<",\"superseded_after_deadline\":"<<timed_out_
     <<",\"pending_at_stop\":"<<at_stop_<<",\"background_contexts\":"<<background_
     <<",\"cache_lookup_ms\":"<<double(lookup_ns_)/1e6<<",\"context_prepare_ms\":"<<double(prepare_ns_)/1e6
     <<",\"context_verify_ms\":"<<double(verify_ns_)/1e6
     <<",\"expired_pending\":"<<expired_pending_<<",\"tf_rows\":"<<tf_rows_<<",\"tf_calls\":"<<tf_calls_
     <<",\"mixed_tf_calls\":"<<mixed_calls_<<",\"mixed_tf_cuda_ms\":"<<mixed_cuda_ms_
     <<",\"tf_cuda_ms\":"<<tf_cuda_ms_
     <<",\"max_tf_roundtrip_ms\":"<<double(max_tf_roundtrip_ns_)/1e6
     <<",\"tf_phases\":{";
  constexpr std::array names={"cold","refill","steady"};
  for(size_t i=0;i<Phases;++i)out<<(i?",":"")<<'"'<<names[i]<<"\":{\"rows\":"<<phases_[i].rows
   <<",\"cuda_ms_by_row_share\":"<<phases_[i].cuda_ms
   <<",\"exclusive_calls\":"<<phases_[i].exclusive_calls
   <<",\"exclusive_cuda_ms\":"<<phases_[i].exclusive_ms<<'}';
  out<<"},\"tf_batch_histogram\":[";
  for(size_t i=1;i<=Batch;++i)out<<(i==1?"":",")<<batch_sizes_[i];
  out<<']'
     <<",\"max_concurrent_decisions\":"<<peak_batch<<",\"pending_observation_peak\":"<<high_water<<",\"slots\":[";
  for(size_t i=0;i<pending_.size();++i){
   out<<(i?",":"")<<"["<<gates_[i].accepted<<','<<deferred_[i]<<','<<warm_deferred_[i]<<"]";
  }
  out<<"]}\n";
  return out.str();
 }

 template<class ReadClock,class Stop,class Emit,class Progress>
 void run(int fd,ReadClock clock,Stop stop,Emit emit,Progress progress){
  std::vector<Decision> decisions;decisions.reserve(Batch);
  std::vector<Context> contexts;contexts.reserve(Batch);
  std::vector<Candidate> candidates;candidates.reserve(stage1::SmallCapSlots);
  auto inputs=std::make_unique<std::array<pure::v2::PreparedInput,Batch>>();
  std::array<TF,Batch> outputs;
  size_t emitted=0;
  try{
   while(!stop()){
    progress(0);
    if(!depth){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
    auto state=clock();
    if(!state.valid){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
    const auto lookup=integrated::now();
    candidates.clear();contexts.clear();decisions.clear();emitted=0;
    std::array<Snapshot,stage1::SmallCapSlots> snapshots;
    {
     std::lock_guard lock(mutex_);
     for(size_t slot=0;slot<pending_.size();++slot){
      auto& lane=pending_[slot];
      while(!lane.empty()){
       const auto& job=lane.front();
       // Publication and source recovery can advance after the clock read.
       if(job->enqueued>state.mono||job->id.generation>state.generation||state.wall<job->id.bar)break;
       if(job->id.generation==state.generation&&state.wall-job->id.bar<=5*NS&&
          state.mono-job->eligible<NS){
        snapshots[slot]=job;
        break;
       }
       defer(*job);lane.pop_front();--depth;++expired_pending_;
      }
     }
    }
    for(const auto& job:snapshots){
     if(!job||job->id.generation!=state.generation||state.wall<job->id.bar||state.wall-job->id.bar>5*NS)continue;
     auto deadline=std::min(job->eligible+NS,state.mono+(job->id.bar+5*NS-state.wall));
     candidates.push_back({job,misses(*job),deadline});
    }
    // Ready decisions go first. For remaining candidates, prioritize a
    // feasible completion by earliest deadline. Use free rows for priming.
    const auto tf_budget=std::max(TransformerBudget,PostTransformerReserve+
                                 *std::max_element(recent_tf_roundtrip_ns_.begin(),recent_tf_roundtrip_ns_.end()));
    auto rank=[&](const Candidate& c){
     auto count=c.missing.count();
     auto budget=count?tf_budget:SacBudget;
     auto kind=c.deadline<=state.mono+budget?2:count==0?0:count<=Batch?1:2;
     // Within a 200 ms deadline band, use scarce rows for more
     // decisions. Older candidates still move ahead of cheap new work.
     const auto band=(c.deadline-state.mono)/(NS/5);
     return std::tuple(kind,kind==2?int64_t(count):band,kind==1?count:size_t(0),
                       served_[c.job->id.symbol],c.deadline,c.job->id.symbol);
    };
    std::sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b){return rank(a)<rank(b);});
    lookup_ns_+=integrated::now()-lookup;

    auto claim=[&](const Candidate& c){
     if(decisions.size()==Batch)return false;
     decisions.emplace_back();
     if(!take(c.job,decisions.back(),integrated::now())){decisions.pop_back();return false;}
     decisions.back().outcome.tf_computed=uint32_t(c.missing.count());
     return true;
    };
    for(const auto& c:candidates)
     if(std::get<0>(rank(c))==0)claim(c);
    if(decisions.empty()){
     for(auto& c:candidates){
      const auto count=c.missing.count();
      if(std::get<0>(rank(c))!=1||count>Batch-contexts.size()||!claim(c))continue;
      for(size_t i=0;i<History;++i)if(c.missing[i])contexts.push_back({c.job,i,!primed(c.job->id.symbol,c.job->id.generation)?Cold:count==1?Steady:Refill});
      c.missing.reset();
     }
     for(const auto& c:candidates){
      const bool cold=!primed(c.job->id.symbol,c.job->id.generation);
      // A superseding bar keeps the newest contexts in its window longer.
      // Spend spare rows there before refilling the oldest edge.
      // An expired observation can only warm a future, strictly newer bar.
      const size_t oldest=c.deadline<=state.mono+SacBudget?1:0;
      for(size_t i=History;i>oldest&&contexts.size()<Batch;--i)if(c.missing[i-1]){
       contexts.push_back({c.job,i-1,cold?Cold:Refill});++background_;
      }
      if(contexts.size()==Batch)break;
     }
    }
    if(decisions.empty()&&contexts.empty()){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
    peak_batch=std::max(peak_batch,decisions.size());
    progress(decisions.size());
    integrated::Reply tf_reply;
    if(!contexts.empty()){
     const auto prep=integrated::now();
     std::array<size_t,Phases> rows{};
     for(size_t i=0;i<contexts.size();++i){
      (*inputs)[i]=contexts[i].job->history.context(contexts[i].index);
      ++rows[contexts[i].phase];
     }
     prepare_ns_+=integrated::now()-prep;
     for(auto& d:decisions)d.outcome.prepared=integrated::now();
     const auto count=contexts.size();
     integrated::Header h;h.sequence=++wire_;h.identity=h.sequence;h.count=uint32_t(count);
     integrated::transfer(fd,&h,sizeof(h),true);
     integrated::transfer(fd,inputs->data(),count*sizeof((*inputs)[0]),true);
     integrated::transfer(fd,&tf_reply,sizeof(tf_reply),false);integrated::validate(h,tf_reply);
     integrated::transfer(fd,outputs.data(),count*sizeof(TF),false);
     const auto roundtrip=integrated::now()-prep;
     recent_tf_roundtrip_ns_[tf_calls_%recent_tf_roundtrip_ns_.size()]=roundtrip;
     max_tf_roundtrip_ns_=std::max(max_tf_roundtrip_ns_,roundtrip);
     tf_rows_+=count;++tf_calls_;++batch_sizes_[count];tf_cuda_ms_+=tf_reply.compute_ms;
     const bool mixed=std::count_if(rows.begin(),rows.end(),[](auto n){return n>0;})>1;
     if(mixed){++mixed_calls_;mixed_cuda_ms_+=tf_reply.compute_ms;}
     for(size_t i=0;i<Phases;++i){
      auto& phase=phases_[i];
      phase.rows+=rows[i];phase.cuda_ms+=tf_reply.compute_ms*rows[i]/count;
      if(rows[i]==count){++phase.exclusive_calls;phase.exclusive_ms+=tf_reply.compute_ms;}
     }
     if(capture_){capture_->tf_contexts+=count;++capture_->tf_batches;capture_->tf_cuda_ms+=tf_reply.compute_ms;}
     for(size_t i=0;i<contexts.size();++i){
      const auto& c=contexts[i];const auto& id=c.job->id;
      entry(*c.job,c.index)={end(*c.job,c.index),id.session,id.epoch,id.generation,id.execution_epoch,(*inputs)[i],outputs[i]};
     }
     for(const auto& c:contexts)if(!primed(c.job->id.symbol,c.job->id.generation)&&misses(*c.job).none()){
      primed_[c.job->id.symbol]=c.job->id.generation+1;
     }
    }
    for(auto& d:decisions){
     auto& r=d.outcome;
     if(!r.prepared)r.prepared=integrated::now();
     r.tf_completed=integrated::now();r.tf_ms=tf_reply.compute_ms;
     r.tf_transfer_ms=tf_reply.transfer_ms;r.tf_materialize_ms=tf_reply.materialize_ms;
     const auto verification=integrated::now();
     std::array<TF,History> history;
     std::array<double,History> prices;
     for(size_t i=0;i<History;++i){
      require(cached(*d.job,i),"microbatch context identity missing");
      const auto& context=entry(*d.job,i);
      const auto expected=d.job->history.context(i);
      require(!std::memcmp(&expected,&context.input,sizeof(expected)),"microbatch cached context content changed");
      history[i]=context.value;
      prices[i]=d.job->history.model.labeled.back(History-1-i).bar.close;
     }
     verify_ns_+=integrated::now()-verification;
     d.observation=builder_.build(history,prices);r.sac_prepared=integrated::now();
    }
    // Reuse the production SAC wire batch of up to four. Reject stale or
    // expired work before submitting it, and recheck after the reply.
    while(emitted<decisions.size()){
     std::array<std::array<float,490>,4> obs{};
     std::array<size_t,4> index{};
     size_t logical=0, next=emitted;
     for(;next<decisions.size()&&logical<obs.size();++next){
      auto& d=decisions[next];auto& r=d.outcome;auto current=clock();
      if(!current.valid||current.generation!=r.id.generation||current.wall<r.id.bar||current.wall-r.id.bar>5*NS)
       r.disposition=7;
      else if(integrated::expired(r.eligible,current.mono))r.disposition=2;
      else{index[logical]=next;obs[logical++]=d.observation;}
      if(r.disposition){r.completed=current.mono;r.source_age_ms=double(current.wall-r.id.bar)/1e6;}
     }
     if(logical){
      integrated::Header h;h.sequence=++wire_;h.identity=h.sequence;h.stage=1;h.count=uint32_t(logical);
      integrated::transfer(fd,&h,sizeof(h),true);
      integrated::transfer(fd,obs.data(),logical*sizeof(obs[0]),true);
      integrated::Reply reply;
      integrated::transfer(fd,&reply,sizeof(reply),false);integrated::validate(h,reply);
      std::array<float,4> actions{};
      integrated::transfer(fd,actions.data(),logical*sizeof(float),false);
      if(capture_){capture_->sac_rows+=logical;++capture_->sac_calls;capture_->sac_cuda_ms+=reply.compute_ms;}
      for(size_t i=0;i<logical;++i){
       auto& d=decisions[index[i]];auto& r=d.outcome;
       r.sac_completed=integrated::now();r.action=actions[i];
       r.sac_transfer_ms=reply.transfer_ms;r.sac_ms=reply.compute_ms;
       r.target=integrated::flat_target(r.action,d.job->history.model.labeled.back(0).bar.close);
       auto current=clock();r.completed=current.mono;r.source_age_ms=double(current.wall-r.id.bar)/1e6;
       if(!current.valid||current.generation!=r.id.generation||current.wall<r.id.bar||current.wall-r.id.bar>5*NS)
        r.disposition=7;
       else if(integrated::expired(r.eligible,r.completed))r.disposition=3;
       if(!r.disposition){++served_[r.id.symbol];if(capture_)capture_->record(r.id.job,d.observation);}
      }
     }
     while(emitted<next){emit(decisions[emitted].outcome);++emitted;}
    }
    decisions.clear();emitted=0;
   }
   integrated::Header h;h.stage=2;h.count=1;
   integrated::transfer(fd,&h,sizeof(h),true);
   std::array<char,8> ack{};integrated::transfer(fd,ack.data(),ack.size(),false);
   require(std::string_view(ack.data(),ack.size())=="STOP_ACK","microbatch GPU stop acknowledgement invalid");
  }catch(...){
   for(;emitted<decisions.size();++emitted){
    auto& r=decisions[emitted].outcome;if(!r.id.job)continue;r.disposition=4;r.completed=integrated::now();emit(r);
   }
   throw;
  }
 }
};

}
