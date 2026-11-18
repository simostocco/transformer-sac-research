#pragma once
#include "core.hpp"
#include "pinned_flat_pipeline.hpp"
namespace shadow {
using Digest256=pure::v2::Digest;
using SnapshotIdentity=pure::v2::SnapshotId;
using ModelRevisionIdentity=pure::v2::ModelRevisionIdentity;
using ObservationDigest=pure::v2::ObservationDigest;
using PureInferenceRequest=pure::v2::Request;
using PureInferenceResult=pure::v2::Result;
using PureJobStatus=pure::v2::PureJobStatus;
inline constexpr size_t TRANSFORMER_QUANTILE_DIM=pure::v2::QuantileDim;
inline constexpr size_t TRANSFORMER_EMBEDDING_DIM=pure::v2::EmbeddingDim;
inline constexpr size_t TRANSFORMER_RESULT_FLOAT_DIM=pure::v2::TransformerResultDim;
inline constexpr size_t SAC_ACTION_DIM=pure::v2::SacActionDim,STATS_DIM=pure::v2::SacStatsDim;
inline SnapshotIdentity inference_identity(const SnapshotId& s){return {s.session,s.source_epoch,s.execution_epoch,s.generation,s.model,s.frontier,s.symbol,s.end};}
inline pure::v2::SourceCut inference_cut(const SourceCut& c){return {c.source_sequence,c.finalization_sequence,c.outer,c.feed_ns};}
inline void validate_result(const PureInferenceRequest& q,const PureInferenceResult& r){
 require(q.abi_version==r.abi_version&&q.key==r.key&&q.revisions.transformer==r.revisions.transformer&&q.revisions.sac==r.revisions.sac&&q.input_digest==r.input_digest&&pure::v2::same_snapshot(q.snapshot,r.snapshot)&&pure::v2::same_cut(q.source_cut,r.source_cut)&&q.owner_contract==r.owner_contract&&q.flat_owner_revision==r.flat_owner_revision,"pure result identity mismatch");
 if(r.status==pure::v2::Status::Ready){
  for(float x:r.transformer_output)require(std::isfinite(x),"nonfinite Transformer result");
  require(bool(r.sac_valid)==(q.owner_contract==pure::v2::OwnerContract::PinnedFlatV1),"SAC purity contract");
  if(r.sac_valid){require(q.flat_owner_revision&&pure::v2::present(r.observation_digest),"unattested SAC result");for(float x:r.sac_stats)require(std::isfinite(x),"nonfinite stats");require(std::isfinite(r.sac_action[0])&&std::abs(r.sac_action[0])<=1&&pure::v2::same_float(r.sac_action[0],r.sac_stats[6]),"invalid SAC action");}
 }
}
}
