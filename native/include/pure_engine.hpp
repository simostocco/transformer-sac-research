#pragma once
#include "inference_interface.hpp"
#include "model_state.hpp"
#include "attestation.hpp"
#include <span>
#include "early_input_publication.hpp"
namespace shadow {
// An implementation supplies only model data. No execution token or PAPER owner
// crosses this boundary. False means unsupported exact evidence, never fallback.
struct ModelBackend {
 virtual ~ModelBackend()=default;
 virtual bool transformer(const PureInferenceRequest&,std::array<float,439>&)=0;
 virtual bool observation(const PureInferenceRequest&,const std::array<float,439>&,pure::v2::FlatObservation&)=0;
 virtual bool sac(const pure::v2::FlatObservation&,std::array<float,1>&,std::array<float,17>&)=0;
};
struct ModelOwnerConfiguration {
 pure::v2::OwnerContract contract=pure::v2::OwnerContract::MarketOnly;
 uint64_t revision=0;
 bool position_dependent=false,pending_dependent=false,account_dependent=false;
 uint64_t validate()const{require(contract==pure::v2::OwnerContract::MarketOnly||contract==pure::v2::OwnerContract::PinnedFlatV1,"unknown owner contract");if(contract==pure::v2::OwnerContract::PinnedFlatV1)require(revision&&!position_dependent&&!pending_dependent&&!account_dependent,"PINNED_FLAT_MODEL_OWNER required");else require(!revision,"market-only owner revision");return revision;}
};
struct ExactOracleRecord {PureInferenceRequest request{};pure::v2::FlatObservation observation{};PureInferenceResult result{};};
// Storage is loaded once by the owner and remains immutable throughout service.
class PrecomputedBackend final:public ModelBackend {
 std::span<const ExactOracleRecord> records;
 const ExactOracleRecord* find(const PureInferenceRequest& q)const{for(const auto& r:records)if(r.request.key==q.key){validate_result(q,r.result);require(std::memcmp(&q.input,&r.request.input,sizeof(q.input))==0,"oracle input bytes mismatch");return &r;}return nullptr;}
public:
 explicit PrecomputedBackend(std::span<const ExactOracleRecord> r):records(r){}
 bool transformer(const PureInferenceRequest& q,std::array<float,439>& out)override{auto r=find(q);if(!r)return false;out=r->result.transformer_output;return true;}
 bool observation(const PureInferenceRequest& q,const std::array<float,439>& tf,pure::v2::FlatObservation& out)override{auto r=find(q);if(!r||!r->result.sac_valid)return false;require(tf==r->result.transformer_output,"oracle TF mismatch");out=r->observation;return true;}
 bool sac(const pure::v2::FlatObservation& o,std::array<float,1>& a,std::array<float,17>& s)override{for(const auto& r:records)if(r.request.key==o.key){require(r.result.sac_valid&&o.observation_digest==r.result.observation_digest&&std::memcmp(o.values.data(),r.observation.values.data(),sizeof(o.values))==0,"oracle observation mismatch");a=r.result.sac_action;s=r.result.sac_stats;return true;}return false;}
};
inline Digest256 input_digest(const pure::v2::PreparedInput& input){AdmissionMac zero{};return admission_mac(zero,reinterpret_cast<const uint8_t*>(&input),sizeof(input));}
// Same in-process HMAC-SHA256 attestation as B; padding is never hashed.
inline Digest256 exact_key(const PureInferenceRequest& r){
 std::array<uint64_t,12> words={r.snapshot.session,r.snapshot.source_epoch,r.snapshot.execution_epoch,r.snapshot.generation,r.snapshot.model,r.snapshot.frontier,r.snapshot.symbol,uint64_t(r.snapshot.end_ns),r.source_cut.source_sequence,r.source_cut.finalization_sequence,r.source_cut.outer,uint64_t(r.source_cut.feed_ns)};
 std::array<uint8_t,96+96+8> bytes{};size_t n=0;for(auto w:words)for(size_t j=0;j<8;++j)bytes[n++]=uint8_t(w>>(8*j));
 for(auto* d:{&r.revisions.transformer,&r.revisions.sac,&r.input_digest})for(auto v:*d)bytes[n++]=v;
 for(size_t j=0;j<8;++j)bytes[n++]=uint8_t(r.flat_owner_revision>>(8*j));
 return admission_mac({},bytes.data(),n);
}
// One event-loop owner. Events are synchronous callbacks; polling observes only
// terminal results. Consumers retain a B registry reference until acknowledgement.
// An immutable input COPY lives in C's bounded slot, not the mutable state rings.
template<size_t Capacity=40,size_t Symbols=320> class NativePureEngine {
 pure::v2::PinnedFlatPipeline<Capacity> pipeline;
 struct Slot {bool used=false;SnapshotId id{};Digest256 key{};};std::array<Slot,Capacity> slots{};
 static SnapshotId b_id(const pure::v2::SnapshotId& s){return {s.session,s.source_epoch,s.execution_epoch,s.generation,s.model,s.frontier,s.symbol,s.end_ns};}
 Slot* find(const SnapshotId& id){for(auto& s:slots)if(s.used&&s.id==id)return &s;return nullptr;}
public:
 Jobs<Capacity,Symbols> registry;
 static_assert(Capacity<=40,"B input credit is forty pending jobs");
 explicit NativePureEngine(uint64_t flat_revision=0):pipeline(flat_revision){}
 explicit NativePureEngine(const ModelOwnerConfiguration& config):pipeline(config.validate()){}
 bool publish(const PureInferenceRequest& q){
  require(q.snapshot.session&&q.snapshot.source_epoch&&q.snapshot.execution_epoch&&q.snapshot.generation&&q.snapshot.model&&q.snapshot.frontier,"incomplete native snapshot identity");
  auto id=b_id(q.snapshot);auto old=find(id);
  for(const auto& slot:slots)require(!slot.used||slot.key!=q.key||slot.id==id,"key reused for different snapshot");
  require(!old||old->key==q.key,"one PureJob per snapshot");
  if(old){bool accepted=pipeline.submit(q);if(accepted)registry.promote_exact(id,uint8_t(q.priority));return accepted;}
  auto slot=std::find_if(slots.begin(),slots.end(),[](const Slot& s){return !s.used;});if(slot==slots.end())return false;
  // ticket enforces B's irreversible retirement floors before model submission.
  if(!pipeline.submit(q))return false;
  try{registry.ticket(id);}catch(...){pipeline.fail(q.key,3);pipeline.acknowledge(q.key);throw;}
  registry.publish(id,1,q.source_cut.finalization_sequence,1,false);registry.promote_exact(id,uint8_t(q.priority));*slot={true,id,q.key};return true;
 }
 void admit(PureInferenceRequest q,uint64_t command,int64_t deadline){q.priority=pure::v2::Priority::P0;require(publish(q),"pure input backpressure");registry.admit(b_id(q.snapshot),command,deadline);}
 bool promote(PureInferenceRequest q){q.priority=pure::v2::Priority::P1;return publish(q);}
 void release(uint64_t command){registry.release(command);}
 const PureInferenceResult* result(const SnapshotId& id){auto s=find(id);return s?pipeline.result(s->key):nullptr;}
 auto dispatch(){auto batch=pipeline.dispatch();if(batch)for(size_t n=0;n<batch->logical_count;++n){auto& q=pipeline.request(batch->logical[n]);if(batch->stage==pure::v2::Stage::Transformer)registry.start_exact(b_id(q.snapshot));}return batch;}
 template<class Complete> bool service(ModelBackend& backend,Complete notify){
  auto batch=dispatch();if(!batch)return false;
  for(size_t n=0;n<batch->logical_count;++n){auto slot=batch->logical[n];const auto& q=pipeline.request(slot);auto id=b_id(q.snapshot);bool supported=false;uint32_t error=1;
   try{
   if(batch->stage==pure::v2::Stage::Transformer){std::array<float,439> tf{};supported=backend.transformer(q,tf);if(supported){pipeline.transformer_complete(slot,tf);if(q.owner_contract==pure::v2::OwnerContract::PinnedFlatV1){pure::v2::FlatObservation o;supported=backend.observation(q,tf,o);if(supported)pipeline.observation_ready(o);}}}
   else {std::array<float,1>a{};std::array<float,17>s{};supported=backend.sac(pipeline.observation(slot),a,s);if(supported)pipeline.sac_complete(slot,a,s);}
   }catch(const std::exception&){supported=false;error=4;}
   if(!supported){pipeline.fail(q.key,error);registry.fail(id);} // error 1: unsupported exact oracle
   if(auto r=pipeline.result(q.key)){validate_result(q,*r);if(r->status==pure::v2::Status::Ready)registry.complete(id,r->sac_valid?r->sac_action[0]:r->transformer_output[0]);notify(id,*r);}
  }return true;
 }
 void retire(const SnapshotId& id){auto s=find(id);require(s,"missing exact job");auto e=registry.inspect(id);require(!e.refs&&(e.state==JobState::READY||e.state==JobState::FAILED),"referenced/nonterminal retire");pipeline.acknowledge(s->key);registry.retire_exact(id);s->used=false;}
 void close(){registry.close();for(auto& s:slots)if(s.used&&!pipeline.result(s.key)){pipeline.fail(s.key,2);registry.fail(s.id);}}
};
// Called immediately after the canonical owner appends feature/model state.
// No preparation queue; incomplete history emits nothing.
template<class Clock,class Emit>
bool publish_model_ready(const SnapshotId& id,const SourceCut& cut,const ModelFacingState& model,pure::v2::ReadinessTimes times,Clock clock,Emit emit){
 if(model.labeled.size<192||model.ready.size<183)return false;
 require(model.labeled.back().bar.ts==id.end,"snapshot/state cut mismatch");
 PureInferenceRequest q;q.snapshot=inference_identity(id);q.source_cut=inference_cut(cut);q.revisions.transformer=pure::v2::TransformerRevision;
 auto context=model.context();
 pure::v2::publish_exact_input(q,context,id.end,times,input_digest,clock,[&](pure::v2::ExactModelInputReady event){event.request.key=exact_key(event.request);emit(event);});return true;
}
}
