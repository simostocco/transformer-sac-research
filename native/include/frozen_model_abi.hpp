#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <type_traits>
namespace pure::v2 {
inline constexpr uint32_t AbiVersion=2;
inline constexpr size_t Sequence=120, FeatureWidth=31, QuantileTargets=11, QuantilesPerTarget=5;
inline constexpr size_t QuantileDim=55, EmbeddingDim=384, TransformerResultDim=439;
inline constexpr size_t SacObservationDim=490, SacActionDim=1, SacStatsDim=17, PhysicalBatch=20;
using Digest=std::array<uint8_t,32>;
using ObservationDigest=Digest;
struct ModelRevisionIdentity {Digest transformer{},sac{};};
enum class PureJobStatus:uint8_t {ABSENT,QUEUED,RUNNING,READY,FAILED,RETIRED};
constexpr Digest parse_digest(const char* s){Digest out{};for(size_t i=0;i<32;++i){auto value=[](char c){return c>='a'?c-'a'+10:c-'0';};out[i]=uint8_t(value(s[2*i])*16+value(s[2*i+1]));}return out;}
inline constexpr Digest TransformerRevision=parse_digest("a31bd58c846a04d8da6d7f86f48ca7646c9adda7285f36e7ab1ca0ebe14c8d71");
inline constexpr Digest SacRevision=parse_digest("00e1d33da2a1e2fda619415b97b2467d87d1c2815ab03d036459cfa59ef68046");
// Copied explicitly from Agent B coordinates; no reinterpret_cast ABI coupling.
struct SnapshotId {uint64_t session=0,source_epoch=0,execution_epoch=0,generation=0,model=0,frontier=0;uint32_t symbol=0;int64_t end_ns=0;};
struct SourceCut {uint64_t source_sequence=0,finalization_sequence=0,outer=0;int64_t feed_ns=0;};
struct PreparedInput {std::array<float,Sequence*FeatureWidth> x{};std::array<int64_t,Sequence> k1{},k2{};std::array<float,Sequence> relative{};};
enum class Priority:uint8_t { P0,P1,P2 };
enum class OwnerContract:uint8_t { MarketOnly,PinnedFlatV1 };
enum class Stage:uint8_t { Transformer,SAC };
enum class Status:uint8_t { Pending,Ready,Failed };
struct Request {
 uint32_t abi_version=AbiVersion;
 Digest key{};ModelRevisionIdentity revisions{};Digest input_digest{};
 SnapshotId snapshot{};SourceCut source_cut{};PreparedInput input{};
 uint64_t deadline_monotonic_ns=0,flat_owner_revision=0;
 Priority priority=Priority::P2;OwnerContract owner_contract=OwnerContract::MarketOnly;
};
struct FlatObservation {
 Digest key{};ObservationDigest observation_digest{};Digest history_digest{};
 SnapshotId snapshot{};SourceCut source_cut{};
 uint64_t flat_owner_revision=0;
 uint32_t history_rows=0;int64_t first_history_end_ns=0,last_history_end_ns=0;
 std::array<float,SacObservationDim> values{};
};
// Production _materialize_output exposes 55 q values, checkpoint order, then
// NumPy mean16 pooled 384-value embedding. Auxiliary edge[4]/direction[2]
// are not production loc_* outputs and are deliberately not appended here.
struct Result {
 uint32_t abi_version=AbiVersion;
 Digest key{};ModelRevisionIdentity revisions{};Digest input_digest{};ObservationDigest observation_digest{};
 SnapshotId snapshot{};SourceCut source_cut{};uint64_t flat_owner_revision=0;
 std::array<float,TransformerResultDim> transformer_output{};
 std::array<float,SacActionDim> sac_action{};
 // Exact shared _materialize_sac_stats insertion order. Keeping all diagnostics
 // avoids dropping existing policy fields while the executable action is dim 1.
 std::array<float,SacStatsDim> sac_stats{};
 OwnerContract owner_contract=OwnerContract::MarketOnly;
 Status status=Status::Pending;uint8_t sac_valid=0;uint32_t error_code=0;
};
// In-process notifications are synchronous on the single owner. Callbacks may
// copy fixed descriptors; a command reference pins terminal storage until release.
struct PureCompletionNotification {Digest key{};SnapshotId snapshot{};PureJobStatus status=PureJobStatus::ABSENT;uint32_t error_code=0;uint64_t sequence=0;};
static_assert(sizeof(Request)==17536 && sizeof(Result)==2112);
static_assert(sizeof(SnapshotId)==64 && sizeof(ModelRevisionIdentity)==64);
static_assert(offsetof(Request,input)==232 && offsetof(Result,transformer_output)==272);
inline constexpr std::array<const char*,SacStatsDim> SacStatNames={
 "z_det","z_raw_signed","action_det_exec","action_det_signed","action_raw_signed",
 "action_det_buy01","action_exec_signed","beta_alpha","beta_beta","beta_mean01",
 "beta_conc","inv","sell_room","buy_room","sell_limit","buy_limit","transform_gap"};
// SAC sorts loc quantile column names lexically. Transformer emits checkpoint
// target_spec order. Generated/checked by audit_frozen_contract.py.
inline constexpr std::array<size_t,QuantileDim> SacQuantileOrder={
 5,6,7,8,9,15,16,17,18,19,25,26,27,28,29,35,36,37,38,39,
 0,1,2,3,4,10,11,12,13,14,20,21,22,23,24,30,31,32,33,34,
 40,41,42,43,44,45,46,47,48,49,50,51,52,53,54};
inline constexpr std::array<size_t,10> FlatZeroIndices={95,96,97,98,100,101,102,103,104,105};
static_assert(QuantileTargets*QuantilesPerTarget+EmbeddingDim==TransformerResultDim);
static_assert(std::is_standard_layout_v<Result>&&std::is_trivially_copyable_v<Result>);
static_assert(std::is_trivially_copyable_v<Request>);
static_assert(sizeof(PreparedInput)==17280);
}
