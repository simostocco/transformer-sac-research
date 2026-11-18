#pragma once
#include "pinned_flat_pipeline.hpp"
namespace pure::v2 {
struct ReadinessTimes {
 uint64_t canonical_closed_ns=0,feature_ready_ns=0,history_ready_ns=0,context_ready_ns=0,
          exact_input_ready_ns=0,pure_job_created_ns=0,pure_job_dispatched_ns=0,pure_job_completed_ns=0,command_admission_ns=0;
};
struct ExactModelInputReady {Request request{};ReadinessTimes times{};};
struct SourcePreparationTicket {Digest key{};SnapshotId snapshot{};SourceCut source_cut{};Priority priority=Priority::P2;};
// Pending *source* work, not a second feature owner. No row may be skipped.
// Agent B retains the canonical row bytes indexed by SnapshotId/SourceCut.
template<size_t Capacity=256,size_t Symbols=320> class InputPreparationQueue {
 struct Item {bool used=false;SourcePreparationTicket identity{};uint64_t source_committed_ns=0,order=0;};
 std::array<Item,Capacity> items_{};uint64_t order_=0;size_t inflight_=Capacity;
 struct Cursor {bool used=false;SourcePreparationTicket source{};int64_t last_enqueued_ns=0;};
 std::array<Cursor,Symbols> cursors_{};
 static bool same_source(const SourcePreparationTicket&a,const SourcePreparationTicket&b){return a.snapshot.session==b.snapshot.session&&a.snapshot.source_epoch==b.snapshot.source_epoch&&a.snapshot.symbol==b.snapshot.symbol;}
public:
 bool enqueue(const SourcePreparationTicket& metadata,uint64_t source_committed_ns){
  for(auto&i:items_)if(i.used&&i.identity.key==metadata.key)throw std::invalid_argument("duplicate source preparation row");
  auto slot=std::find_if(items_.begin(),items_.end(),[](const Item&i){return !i.used;});if(slot==items_.end())return false;
  auto cursor=std::find_if(cursors_.begin(),cursors_.end(),[&](const Cursor&c){return c.used&&same_source(c.source,metadata);});
  if(cursor!=cursors_.end()&&metadata.snapshot.end_ns!=cursor->last_enqueued_ns+1000000000LL)throw std::invalid_argument("source preparation gap or reorder");
  if(cursor==cursors_.end())cursor=std::find_if(cursors_.begin(),cursors_.end(),[](const Cursor&c){return !c.used;});
  if(cursor==cursors_.end())return false;
  cursor->used=true;cursor->source=metadata;cursor->last_enqueued_ns=metadata.snapshot.end_ns;
  slot->used=true;slot->identity=metadata;slot->source_committed_ns=source_committed_ns;slot->order=++order_;return true;
 }
 bool enqueue(const Request& r,uint64_t ns){return enqueue(SourcePreparationTicket{r.key,r.snapshot,r.source_cut,r.priority},ns);}
 bool demand(const Digest& key){for(auto&i:items_)if(i.used&&i.identity.key==key){i.identity.priority=Priority::P0;return true;}return false;}
 const SourcePreparationTicket* take(){
  if(inflight_!=Capacity)return nullptr;
  size_t best=Capacity;Priority best_priority=Priority::P2;
  for(size_t i=0;i<Capacity;++i){if(!items_[i].used)continue;bool predecessor=false;Priority priority=items_[i].identity.priority;
   for(const auto&j:items_)if(j.used&&same_source(j.identity,items_[i].identity)){
    if(j.identity.snapshot.end_ns<items_[i].identity.snapshot.end_ns)predecessor=true;
    priority=std::min(priority,j.identity.priority);
   }
   if(!predecessor&&(best==Capacity||std::tie(priority,items_[i].order)<std::tie(best_priority,items_[best].order))){best=i;best_priority=priority;}
  }
  if(best==Capacity)return nullptr;
  inflight_=best;return &items_[best].identity;
 }
 void acknowledge(const Digest& key){if(inflight_==Capacity||items_[inflight_].identity.key!=key)throw std::logic_error("source preparation acknowledgement");items_[inflight_].used=false;inflight_=Capacity;}
};
// Called by the SOLE per-symbol state writer, after its existing ordered append.
// Context is Agent B's actual frozen ModelFacingState::TransformerContext.
// No command, maintenance tick, later symbol event, model call or PAPER state.
template<class Context,class DigestInput,class Clock,class Emit>
void publish_exact_input(Request request,const Context& context,int64_t actual_state_end_ns,ReadinessTimes times,DigestInput digest_input,Clock clock,Emit emit){
 if(request.snapshot.end_ns!=actual_state_end_ns)throw std::invalid_argument("future or older context substitution");
 if(!(times.canonical_closed_ns&&times.canonical_closed_ns<=times.feature_ready_ns&&times.feature_ready_ns<=times.history_ready_ns&&times.history_ready_ns<=times.context_ready_ns))throw std::invalid_argument("readiness stage chronology");
 for(size_t i=0;i<Sequence;++i){
  std::copy(context.x[i].begin(),context.x[i].end(),request.input.x.begin()+i*FeatureWidth);
  request.input.relative[i]=context.relative[i];request.input.k1[i]=context.k1[i];request.input.k2[i]=context.k2[i];
 }
 auto frozen_digest=digest_input(request.input);
 if(!present(frozen_digest)||(present(request.input_digest)&&request.input_digest!=frozen_digest))throw std::invalid_argument("unattested or changed exact input");
 request.input_digest=frozen_digest;
 times.exact_input_ready_ns=clock();
 if(times.exact_input_ready_ns<times.context_ready_ns)throw std::invalid_argument("input-ready clock domain");
 emit(ExactModelInputReady{request,times});
}
}
