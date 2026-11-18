#pragma once
#include "frozen_model_abi.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <tuple>
namespace pure::v2 {
inline bool same_snapshot(const SnapshotId&a,const SnapshotId&b){return a.session==b.session&&a.source_epoch==b.source_epoch&&a.execution_epoch==b.execution_epoch&&a.generation==b.generation&&a.model==b.model&&a.frontier==b.frontier&&a.symbol==b.symbol&&a.end_ns==b.end_ns;}
inline bool same_cut(const SourceCut&a,const SourceCut&b){return a.source_sequence==b.source_sequence&&a.finalization_sequence==b.finalization_sequence&&a.outer==b.outer&&a.feed_ns==b.feed_ns;}
inline bool present(const Digest& a){return std::any_of(a.begin(),a.end(),[](uint8_t b){return b!=0;});}
inline bool same_float(float a,float b){return std::memcmp(&a,&b,sizeof(float))==0;}
// One single-owner bounded registry and physical lane. No timeout/batch timer.
// Models and the exact static-observation builder are external pure callbacks.
template<size_t Capacity=64> class PinnedFlatPipeline {
 enum class Phase:uint8_t { QueuedTF,RunningTF,AwaitObservation,QueuedSAC,RunningSAC,Terminal };
 struct Job {bool used=false,started=false;uint64_t order=0;Priority priority=Priority::P2;Phase phase=Phase::QueuedTF;Request request{};FlatObservation observation{};Result result{};};
 std::array<Job,Capacity> jobs_{};uint64_t order_=0,flat_revision_=0;
 std::array<size_t,PhysicalBatch> active_{};size_t active_count_=0;
 size_t find(const Digest& key)const{for(size_t i=0;i<Capacity;++i)if(jobs_[i].used&&jobs_[i].request.key==key)return i;return Capacity;}
 void finished(size_t i){for(size_t n=0;n<active_count_;++n)if(active_[n]==i){active_[n]=active_[--active_count_];return;}}
 Job& exact(const Digest& key){auto i=find(key);if(i==Capacity)throw std::logic_error("unknown exact snapshot");return jobs_[i];}
 void terminal(Job& j){j.phase=Phase::Terminal;j.result.status=Status::Ready;}
public:
 struct Dispatch {Stage stage=Stage::Transformer;size_t logical_count=0;std::array<size_t,PhysicalBatch> logical{},physical{};};
 explicit PinnedFlatPipeline(uint64_t pinned_flat_owner_revision=0):flat_revision_(pinned_flat_owner_revision){}
 bool submit(const Request& r){
  if(r.abi_version!=AbiVersion||!present(r.key)||!present(r.input_digest)||r.revisions.transformer!=TransformerRevision||r.snapshot.end_ns<=0||r.source_cut.finalization_sequence<r.source_cut.source_sequence||int(r.priority)>2)throw std::invalid_argument("request identity");
  if(r.owner_contract!=OwnerContract::MarketOnly&&r.owner_contract!=OwnerContract::PinnedFlatV1)throw std::invalid_argument("owner contract");
  if(r.owner_contract==OwnerContract::PinnedFlatV1&&(!flat_revision_||r.flat_owner_revision!=flat_revision_||r.revisions.sac!=SacRevision))throw std::invalid_argument("unpinned flat owner");
  for(auto v:r.input.x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite input");
  for(auto v:r.input.relative)if(!std::isfinite(v))throw std::invalid_argument("nonfinite input");
  auto i=find(r.key);
  if(i!=Capacity){auto&j=jobs_[i];const auto&a=j.request;
   if(!same_snapshot(a.snapshot,r.snapshot)||!same_cut(a.source_cut,r.source_cut)||a.revisions.transformer!=r.revisions.transformer||a.revisions.sac!=r.revisions.sac||a.input_digest!=r.input_digest||a.owner_contract!=r.owner_contract||a.flat_owner_revision!=r.flat_owner_revision||std::memcmp(&a.input,&r.input,sizeof(PreparedInput)))throw std::invalid_argument("immutable key collision");
   j.priority=std::min(j.priority,r.priority);return true;
  }
  for(i=0;i<Capacity;++i)if(!jobs_[i].used)break;
  if(i==Capacity)return false;
  auto&j=jobs_[i];j=Job{};j.used=true;j.order=++order_;j.priority=r.priority;j.request=r;
  j.result.key=r.key;j.result.revisions.transformer=r.revisions.transformer;j.result.revisions.sac=r.revisions.sac;j.result.input_digest=r.input_digest;
  j.result.snapshot=r.snapshot;j.result.source_cut=r.source_cut;j.result.owner_contract=r.owner_contract;j.result.flat_owner_revision=r.flat_owner_revision;
  return true;
 }
 std::optional<Dispatch> dispatch(){
  if(active_count_)return std::nullopt;
  auto queued=[](const Job&j){return j.used&&(j.phase==Phase::QueuedTF||j.phase==Phase::QueuedSAC);};
  size_t best=Capacity;
  for(size_t i=0;i<Capacity;++i)if(queued(jobs_[i])&&(best==Capacity||std::tie(jobs_[i].priority,jobs_[i].order)<std::tie(jobs_[best].priority,jobs_[best].order)))best=i;
  if(best==Capacity)return std::nullopt;
  Dispatch out;out.stage=jobs_[best].phase==Phase::QueuedTF?Stage::Transformer:Stage::SAC;
  // At most 20 selection scans; bounded queue, no batching wait.
  for(size_t n=0;n<PhysicalBatch;++n){
   size_t next=Capacity;
   for(size_t i=0;i<Capacity;++i)if(queued(jobs_[i])&&((jobs_[i].phase==Phase::QueuedTF)==(out.stage==Stage::Transformer))&&(next==Capacity||std::tie(jobs_[i].priority,jobs_[i].order)<std::tie(jobs_[next].priority,jobs_[next].order)))next=i;
   if(next==Capacity)break;
   out.logical[out.logical_count++]=next;jobs_[next].started=true;jobs_[next].phase=out.stage==Stage::Transformer?Phase::RunningTF:Phase::RunningSAC;
  }
  for(size_t n=0;n<PhysicalBatch;++n)out.physical[n]=out.logical[std::min(n,out.logical_count-1)];
  active_count_=out.logical_count;std::copy_n(out.logical.begin(),active_count_,active_.begin());return out;
 }
 const Request& request(size_t slot)const{return jobs_.at(slot).request;}
 const FlatObservation& observation(size_t slot)const{return jobs_.at(slot).observation;}
 void transformer_complete(size_t slot,const std::array<float,TransformerResultDim>& output){
  auto&j=jobs_.at(slot);if(j.phase!=Phase::RunningTF)throw std::logic_error("transformer completion phase");
  for(float v:output)if(!std::isfinite(v))throw std::invalid_argument("nonfinite transformer result");
  j.result.transformer_output=output;
  if(j.request.owner_contract==OwnerContract::PinnedFlatV1)j.phase=Phase::AwaitObservation;else terminal(j);
  finished(slot);
 }
 void observation_ready(const FlatObservation& o){
  auto&j=exact(o.key);const auto&r=j.request;
  if(j.phase!=Phase::AwaitObservation||o.flat_owner_revision!=flat_revision_||o.flat_owner_revision!=r.flat_owner_revision||!same_snapshot(o.snapshot,r.snapshot)||!same_cut(o.source_cut,r.source_cut)||!present(o.observation_digest)||!present(o.history_digest)||o.history_rows!=65||o.last_history_end_ns!=r.snapshot.end_ns||o.first_history_end_ns!=r.snapshot.end_ns-64000000000LL)throw std::invalid_argument("exact flat observation lineage");
  for(float v:o.values)if(!std::isfinite(v))throw std::invalid_argument("nonfinite observation");
  for(size_t i:FlatZeroIndices)if(o.values[i]!=0.0f)throw std::invalid_argument("state-dependent SAC observation");
  if(o.values[99]<=0)throw std::invalid_argument("missing exact price feature");
  for(size_t i=0;i<QuantileDim;++i)if(!same_float(o.values[i],j.result.transformer_output[SacQuantileOrder[i]]))throw std::invalid_argument("quantile producer mismatch");
  for(size_t i=0;i<EmbeddingDim;++i)if(!same_float(o.values[106+i],j.result.transformer_output[QuantileDim+i]))throw std::invalid_argument("embedding producer mismatch");
  j.observation=o;j.result.observation_digest=o.observation_digest;j.phase=Phase::QueuedSAC;j.order=++order_;
 }
 void sac_complete(size_t slot,const std::array<float,SacActionDim>& action,const std::array<float,SacStatsDim>& stats){
  auto&j=jobs_.at(slot);if(j.phase!=Phase::RunningSAC)throw std::logic_error("SAC completion phase");
  for(float v:stats)if(!std::isfinite(v))throw std::invalid_argument("nonfinite SAC stats");
  if(!std::isfinite(action[0])||action[0]<-1||action[0]>1||!same_float(action[0],stats[6]))throw std::invalid_argument("SAC action/stat mismatch");
  j.result.sac_action=action;j.result.sac_stats=stats;j.result.sac_valid=1;terminal(j);finished(slot);
 }
 const Result* result(const Digest& key)const{auto i=find(key);return i==Capacity||jobs_[i].phase!=Phase::Terminal?nullptr:&jobs_[i].result;}
 void fail(const Digest& key,uint32_t error){auto i=find(key);if(i==Capacity||jobs_[i].phase==Phase::Terminal)throw std::logic_error("failure phase");jobs_[i].phase=Phase::Terminal;jobs_[i].result.status=Status::Failed;jobs_[i].result.error_code=error;finished(i);}
 void acknowledge(const Digest& key){auto&j=exact(key);if(j.phase!=Phase::Terminal)throw std::logic_error("acknowledge nonterminal");j.used=false;}
};
}
