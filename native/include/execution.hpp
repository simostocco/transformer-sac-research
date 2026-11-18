#pragma once
#include "paper.hpp"
#include "attestation.hpp"
namespace shadow {
struct ExecutionToken {
 SnapshotId snapshot{};uint64_t command=0,nonce=0,source_sequence=0,finalization_sequence=0,source_generation=0,active_admission=0;
 int64_t admission_ns=0,event_ns=0,feed_ns=0,deadline_ns=0;
 std::array<uint64_t,4> provenance_digest{},reference_digest{};AdmissionMac signature{};
 bool operator==(const ExecutionToken&)const=default;
};
using AdmittedToken=ExecutionToken;
struct CurrentAuthority {
 uint64_t session=0,source_epoch=0,execution_epoch=0,source_generation=0,model_generation=0,admission_sequence=0;
 bool fresh=false,receipt=false,prewatch=false,promoted=false,current_pw=false,workspace=false,reference_current=false,model_ready=false;
 int64_t symbol_event_ns=0;std::array<uint64_t,4> reference_digest{};uint64_t frontier=0,model=0;
};
using ExecutionEpoch=CurrentAuthority;
enum class TokenOutcome:uint8_t {Current,AdmissionNotIssued,SessionCrosslink,GlobalFreshness,PrewatchReleased,PromotedReleased,AdmissionReplaced,WorkerIneligible,ReferenceChanged,SymbolExpired,FutureTime,TriggerExpired};
inline bool retired(TokenOutcome r){return r==TokenOutcome::GlobalFreshness||r==TokenOutcome::PrewatchReleased||r==TokenOutcome::PromotedReleased||r==TokenOutcome::AdmissionReplaced||r==TokenOutcome::SymbolExpired||r==TokenOutcome::TriggerExpired;}
inline const char* token_name(TokenOutcome r){switch(r){
 case TokenOutcome::Current:return "CURRENT";case TokenOutcome::AdmissionNotIssued:return "COMMAND_ADMISSION_NOT_ISSUED_IDENTITY_OR_PROVENANCE_CHANGED";
 case TokenOutcome::SessionCrosslink:return "ACTIVE_ADMISSION_COMMAND_CROSSLINK";case TokenOutcome::GlobalFreshness:return "GLOBAL_FRESHNESS_REVOKED";case TokenOutcome::PrewatchReleased:return "ACTIVE_PREWATCH_RELEASED";case TokenOutcome::PromotedReleased:return "ACTIVE_PROMOTED_RELEASED";case TokenOutcome::AdmissionReplaced:return "ACTIVE_ADMISSION_REPLACED";case TokenOutcome::WorkerIneligible:return "WORKER_EXECUTABLE_INELIGIBLE";case TokenOutcome::ReferenceChanged:return "REFERENCE_CHANGED_AFTER_ADMISSION";case TokenOutcome::SymbolExpired:return "SYMBOL_EVENT_FRESHNESS_EXPIRED";case TokenOutcome::FutureTime:return "STALE_EXECUTABLE_COMMAND";case TokenOutcome::TriggerExpired:return "TRIGGER_TTL_EXPIRED";}return "UNKNOWN";}
// Runtime-scoped HMAC admission proof plus bounded active-reference table.
// Releasing references never invalidates an already-issued admission signature.
// The admission adapter calls issue ONLY after
// the original full scanner/provenance/receipt checks have passed. Every field
// and the complete provenance digest are retained, so modification is fatal
// before any expected-revocation classification. Tokens never go to inference.
template<size_t Commands=128> class TokenAuthority {
 struct Slot{bool used=false;AdmittedToken token{};};std::array<Slot,Commands> issued{};uint64_t nonce=0;AdmissionMac key;
 AdmissionMac sign(const AdmittedToken& t)const{
  std::array<uint64_t,26> words={t.snapshot.session,t.snapshot.source_epoch,t.snapshot.execution_epoch,t.snapshot.generation,t.snapshot.model,t.snapshot.frontier,t.snapshot.symbol,uint64_t(t.snapshot.end),t.command,t.nonce,t.source_sequence,t.finalization_sequence,t.source_generation,t.active_admission,uint64_t(t.admission_ns),uint64_t(t.event_ns),uint64_t(t.feed_ns),uint64_t(t.deadline_ns),t.provenance_digest[0],t.provenance_digest[1],t.provenance_digest[2],t.provenance_digest[3],t.reference_digest[0],t.reference_digest[1],t.reference_digest[2],t.reference_digest[3]};
  std::array<uint8_t,26*8> bytes{};for(size_t i=0;i<words.size();++i)for(size_t j=0;j<8;++j)bytes[i*8+j]=uint8_t(words[i]>>(8*j));return admission_mac(key,bytes.data(),bytes.size());
 }
public:
 TokenAuthority():key(admission_key()){}
 explicit TokenAuthority(AdmissionMac secret):key(secret){}
 TokenAuthority(const TokenAuthority&)=delete;TokenAuthority& operator=(const TokenAuthority&)=delete;
 AdmittedToken issue(AdmittedToken token){require(token.command>0&&token.snapshot.session>0&&token.snapshot.source_epoch>0&&token.snapshot.execution_epoch>0&&token.snapshot.generation>0&&token.source_generation>0&&token.active_admission>0&&token.source_sequence>0&&token.finalization_sequence>0&&token.event_ns<=token.admission_ns&&token.feed_ns<=token.admission_ns&&token.admission_ns-token.event_ns<=5*NS&&token.admission_ns-token.feed_ns<=5*NS,"invalid admission facts");
  require(token.deadline_ns==std::min(token.event_ns,token.feed_ns)+5*NS,"deadline differs from original trigger lifetime");
  for(auto& s:issued){require(!s.used||s.token.command!=token.command,"duplicate token admission");}
  for(auto& s:issued)if(!s.used){require(nonce<UINT64_MAX,"admission nonce exhausted");token.nonce=++nonce;token.signature=sign(token);s={true,token};return token;}
  throw std::logic_error("token bound");
 }
 bool authentic(const AdmittedToken& token)const{if(token.nonce==0||token.nonce>nonce)return false;auto expected=sign(token);uint8_t difference=0;for(size_t i=0;i<32;++i)difference|=expected[i]^token.signature[i];return difference==0;}
 TokenOutcome validate(const AdmittedToken& token,const CurrentAuthority& c,int64_t now)const{
  if(!authentic(token))return TokenOutcome::AdmissionNotIssued;
  if(c.session!=token.snapshot.session)return TokenOutcome::SessionCrosslink;
  bool same_admission=c.fresh&&c.promoted&&c.source_epoch==token.snapshot.source_epoch&&c.execution_epoch==token.snapshot.execution_epoch&&c.source_generation==token.source_generation&&c.admission_sequence==token.active_admission;
  if(!same_admission){if(!c.fresh||!c.receipt)return TokenOutcome::GlobalFreshness;if(!c.prewatch)return TokenOutcome::PrewatchReleased;if(!c.promoted)return TokenOutcome::PromotedReleased;return TokenOutcome::AdmissionReplaced;}
  bool non_age=c.fresh&&c.receipt&&c.prewatch&&c.promoted&&c.current_pw&&c.workspace&&c.reference_current&&c.model_ready&&c.execution_epoch==token.snapshot.execution_epoch&&c.model_generation==token.snapshot.generation&&c.frontier==token.snapshot.frontier&&c.model==token.snapshot.model;
  if(!non_age)return TokenOutcome::WorkerIneligible;
  if(c.reference_digest!=token.reference_digest)return TokenOutcome::ReferenceChanged;
  if(now<c.symbol_event_ns)return TokenOutcome::WorkerIneligible;
  if(now-c.symbol_event_ns>5*NS)return TokenOutcome::SymbolExpired;
  if(now<token.event_ns||now<token.feed_ns)return TokenOutcome::FutureTime;
  if(now>token.deadline_ns)return TokenOutcome::TriggerExpired;
  return TokenOutcome::Current;
 }
 void release(uint64_t command){for(auto& s:issued)if(s.used&&s.token.command==command)s.used=false;}
};
// Epoch/generation context for PAPER observations is distinct from freshness
// authority. A new generation's bar never fills an older pending intent.
template<size_t Symbols=320> struct PaperEpochBridge {
 struct State {bool seen=false,decision=false;uint64_t epoch=0,generation=0,decision_epoch=0,decision_generation=0;int64_t model_row=0,accept_wall=0;};std::array<State,Symbols> states{};
 void identity(uint32_t symbol,uint64_t epoch,uint64_t generation){require(symbol<Symbols,"paper epoch symbol bound");auto& s=states[symbol];require(!s.seen||(epoch>=s.epoch&&generation>=s.generation),"regressive PAPER epoch/generation");}
 void decision(uint32_t symbol,uint64_t epoch,uint64_t generation,int64_t model_row,int64_t accept_wall){identity(symbol,epoch,generation);auto& s=states[symbol];s={true,true,epoch,generation,epoch,generation,model_row,accept_wall};}
 bool bar(uint32_t symbol,uint64_t epoch,uint64_t generation,int64_t row,int64_t final_wall){identity(symbol,epoch,generation);auto& s=states[symbol];bool fill=s.decision&&s.decision_epoch==epoch&&s.decision_generation==generation&&row>s.model_row&&final_wall>s.accept_wall;s.seen=true;s.epoch=epoch;s.generation=generation;return fill;}
};
// One chronological owner calls these methods under its source/financial lock.
// No model work is implemented here. Candidate actions are oracle/pure-result inputs.
template<size_t Symbols=320> class NativeOrderedPaper {
 std::mutex& boundary;TokenAuthority<>& authority;uint64_t completed=0;
 struct TerminalRecord{uint64_t command=0;TokenOutcome outcome=TokenOutcome::Current;};
 Ring<TerminalRecord,4096> terminals;uint64_t floor=0;
 void terminal(uint64_t command,TokenOutcome result){if(terminals.size==4096){TerminalRecord old;terminals.pop(old);floor=std::max(floor,old.command);}terminals.push({command,result});}
public:
 PaperLedger<Symbols> ledger;PaperEpochBridge<Symbols> epochs;
 bool failed=false;uint64_t stale_execution=0,stale_decision=0,stale_intent=0,stale_fill=0;
 NativeOrderedPaper(std::mutex& m,TokenAuthority<>& a,PaperConfig config={}):boundary(m),authority(a),ledger(config){}
 TokenOutcome commit(const AdmittedToken& token,const CurrentAuthority& current,int64_t now,double price,double action,double scale,int64_t model_delta,double timeout,double bar_seconds,double accounting_time,uint32_t date,int minute){std::lock_guard lock(boundary);require(!failed,"PAPER owner requires recovery");
  try{
  if(!authority.authentic(token)){failed=true;return TokenOutcome::AdmissionNotIssued;}
  for(size_t i=0;i<terminals.size;++i)if(terminals.back(i).command==token.command)return terminals.back(i).outcome;
  if(token.command<=completed)return TokenOutcome::Current;
  require(token.command>floor,"COMMAND_TERMINAL_HISTORY_EXPIRED");
  auto result=authority.validate(token,current,now);if(result!=TokenOutcome::Current){if(retired(result))terminal(token.command,result);else failed=true;return result;}
  require(token.command>completed,"ordered command regression");
  epochs.identity(token.snapshot.symbol,token.snapshot.execution_epoch,token.snapshot.generation);
  require(std::isfinite(price)&&price>0&&std::isfinite(action)&&action>=-1&&action<=1&&std::isfinite(scale)&&scale>0&&scale/price<double(INT64_MAX)/2,"invalid pure decision or sizing input");
  int64_t shares=std::max<int64_t>(1,int64_t(std::floor(scale/price)));int64_t delta=int64_t(std::nearbyint(action*double(shares)));
  ledger.submit(token.snapshot.symbol,accounting_time,price,action,delta,shares,model_delta,timeout,bar_seconds,date,minute);epochs.decision(token.snapshot.symbol,token.snapshot.execution_epoch,token.snapshot.generation,token.snapshot.end,now);completed=token.command;terminal(token.command,result);return result;
  }catch(...){failed=true;throw;}
 }
 bool bar(uint32_t symbol,uint64_t epoch,uint64_t generation,int64_t row,int64_t final_wall,double timestamp,double price,double trades,double volume,double participation,uint32_t date,int minute,bool current_bar_authority){std::lock_guard lock(boundary);require(!failed,"PAPER owner requires recovery");try{if(!current_bar_authority)return false;if(!epochs.bar(symbol,epoch,generation,row,final_wall))return false;ledger.fill(symbol,timestamp,price,trades,volume,participation,date,minute);return true;}catch(...){failed=true;throw;}}
};
}
