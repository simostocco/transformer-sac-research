#pragma once
#include "frozen_model_abi.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <utility>
#include <tuple>
namespace shadow {
constexpr int64_t NS=1000000000;
constexpr double NaN=std::numeric_limits<double>::quiet_NaN();
inline void require(bool b,const char* why){if(!b)throw std::logic_error(why);}
template<class T,size_t N> struct Ring {
 std::array<T,N> data{}; size_t size=0,head=0;
 void push(const T& x){data[head]=x;head=(head+1)%N;size=std::min(size+1,N);}
 const T& back(size_t lag=0)const{require(lag<size,"ring lag");return data[(head+N-1-lag)%N];}
 bool pop(T& x){if(!size)return false;x=data[(head+N-size)%N];--size;return true;}
 bool try_push(const T& x){if(size==N)return false;push(x);return true;}
};
struct CanonicalBar {
 int64_t ts=0; double close=NaN,volume=0,ofi=0,buy_v=0,sell_v=0;
 uint64_t trades=0,buy_n=0,sell_n=0; bool forward=false;
};
struct SymbolState {
 int64_t current=-1,last_finalized=-1,last_feed=0; double last_close=NaN; CanonicalBar bucket{};
 uint64_t late=0,out_of_order=0;
 template<class Emit> void close(Emit emit){
  auto row=bucket;row.ts=current;row.forward=!std::isfinite(row.close);
  if(row.forward)row.close=last_close;else last_close=row.close;
  emit(row);last_finalized=current;current+=NS;bucket={};
 }
 template<class Emit> void trade(int64_t event,double price,double size,char side,Emit emit){
  auto sec=event/NS*NS;
  if(sec<=last_finalized){++late;return;}if(current>=0&&sec<current){++out_of_order;return;}
  if(current<0)current=sec;
  while(sec>current)close(emit);
  bucket.close=price;bucket.volume+=size;++bucket.trades;
  if(side=='B'){bucket.buy_v+=size;++bucket.buy_n;bucket.ofi+=size;}
  if(side=='A'){bucket.sell_v+=size;++bucket.sell_n;bucket.ofi-=size;}
 }
 template<class Emit> void tick(int64_t feed,Emit emit){require(feed>=last_feed,"feed regression");last_feed=feed;
  while(current>=0&&current<=feed-6*NS)close(emit);
 }
};
// Fixed symbol IDs must follow lexical symbol order to preserve Python's
// (due, symbol) heap tie-break. Execution generations never reset warm state.
template<size_t Symbols=148> struct CanonicalCore {
 const uint64_t session,source_epoch;
 std::array<SymbolState,Symbols> symbols{};
 std::array<uint64_t,Symbols> history_versions{};
 int64_t last_feed=0;
 CanonicalCore(uint64_t s,uint64_t e):session(s),source_epoch(e){}
 void advance_source(uint64_t s,uint64_t epoch,int64_t feed){
  require(s==session&&epoch==source_epoch,"new source owner required");
  require(feed>=last_feed,"feed regression");last_feed=feed;
 }
 template<class Emit,class Before=std::less<size_t>> void tick(uint64_t s,uint64_t epoch,int64_t feed,Emit emit,Before before={}){
  advance_source(s,epoch,feed);
  std::array<bool,Symbols> changed{};
  for(;;){size_t best=Symbols;int64_t due=INT64_MAX;
   for(size_t i=0;i<Symbols;++i)if(symbols[i].current>=0&&symbols[i].current+6*NS<=feed&&(symbols[i].current+6*NS<due||(symbols[i].current+6*NS==due&&best!=Symbols&&before(i,best)))){best=i;due=symbols[i].current+6*NS;}
   if(best==Symbols)break;
   symbols[best].close([&](auto bar){emit(best,bar);});changed[best]=true;
  }
  for(size_t i=0;i<Symbols;++i)if(changed[i])++history_versions[i];
 }
};
constexpr size_t FeatureCount=36;
struct FeatureRow {std::array<float,FeatureCount> values{};};
struct FeatureConfig {size_t entropy=10,persist=10,slope=10,z=180;double eps=1e-12;};
struct FeatureState {
 FeatureConfig config{};
 Ring<double,304> logp,ratio,abs_ratio,vol_log,buy_n,sell_n,buy_v,sell_v,cum_norm,c1,c2;
 Ring<FeatureRow,192> warm;
 double prev_log=NaN,prev_ret=0,ret_fast=0,ret_slow=0,ofi_fast=0,ofi_slow=0,imb_fast=0,imb_slow=0,ema_vol=0,ema_trades=0;
 double cum_ofi=0,cum_vol=0,sum_ratio=0,sumsq_ratio=0; bool initialized=false;
 FeatureRow append(const CanonicalBar&,int ny_second);
};
struct SnapshotId {uint64_t session=0,source_epoch=0,execution_epoch=0,generation=0,model=0,frontier=0;uint32_t symbol=0;int64_t end=0;
 bool operator==(const SnapshotId&)const=default;
 bool same_namespace(const SnapshotId& b)const{auto a=*this,c=b;a.end=c.end=0;return a==c;}
};
struct ModelInputRow {std::array<float,30> values{};float log_close=0;int64_t k1=0,k2=0;};
struct SourceCut {uint64_t source_sequence=0,finalization_sequence=0,outer=0;int64_t feed_ns=0;bool operator==(const SourceCut&)const=default;};
struct ExecutionPackage {SourceCut cut{};SnapshotId id{};CanonicalBar bar{};Ring<FeatureRow,192> history{};Ring<ModelInputRow,183> model_inputs{};std::array<float,pure::v2::TransformerResultDim> model_result{};uint16_t result_size=0;};
using JobState=pure::v2::PureJobStatus;
enum class Reason:uint8_t{None,VersionLifetime,BackgroundCredit,Shutdown,TriggerExpired,GlobalFreshness,ActiveRevoked,SymbolExpired,EpochRevoked,ModelFailure,HistoryNotReady,PrewatchReleased,PromotedReleased,AdmissionReplaced};
struct ExactPureJob {SnapshotId id{};JobState state=JobState::ABSENT;Reason reason=Reason::None;bool used=false,ticket=false,payload=false,fingerprint_set=false;uint64_t fingerprint=0,order=0,boundary=0,charge=0,ticket_identity=0;uint32_t refs=0,computes=0;uint8_t priority=2;double result=0;};
// One metadata lock; validation and model callbacks always execute outside it.
template<size_t N=256,size_t Namespaces=148> class Jobs {
 struct Command{bool used=false;uint64_t id=0;size_t job=0;int64_t deadline=0;};
 struct Floor{bool used=false;SnapshotId id{};};
 std::array<ExactPureJob,N> entries{};std::array<Command,128> commands{};std::array<Floor,Namespaces> floors{};
 std::mutex mutex;std::condition_variable cv;uint64_t order=0;bool closed=false;
 size_t find(const SnapshotId& id){for(size_t i=0;i<N;++i)if(entries[i].used&&entries[i].id==id)return i;return N;}
 size_t entry(const SnapshotId& id){require(!closed,"registry closed");auto i=find(id);if(i<N)return i;
  for(auto& f:floors)if(f.used&&f.id.same_namespace(id))require(id.end>f.id.end,"version already retired");
  prune_locked(versions_);
  for(i=0;i<N;++i)if(!entries[i].used){entries[i]={};auto& e=entries[i];e.used=true;e.id=id;e.order=++order;return i;}
  throw std::logic_error("job capacity");
 }
 void retire(ExactPureJob& e,Reason why){require(!e.refs&&e.state!=JobState::RUNNING,"retire referenced or running");e.payload=false;e.ticket=false;e.state=JobState::RETIRED;e.reason=why;}
 size_t versions_=192;
 void prune_locked(size_t versions){require(versions>0,"versions");for(auto& e:entries)if(e.used&&!e.refs&&e.state!=JobState::RUNNING){size_t newer=0;for(auto& n:entries)if(n.used&&n.id.same_namespace(e.id)&&n.id.end>e.id.end)++newer;
  if(newer<versions)continue;
  Floor* floor=nullptr;for(auto& f:floors)if(f.used&&f.id.same_namespace(e.id)){floor=&f;break;}
  if(!floor)for(auto& f:floors)if(!f.used){floor=&f;break;}
  require(floor,"namespace capacity");if(!floor->used||floor->id.end<e.id.end)*floor={true,e.id};retire(e,Reason::VersionLifetime);e.used=false;
 }cv.notify_all();}
public:
 void ticket(const SnapshotId& id,uint64_t ticket_identity=1){std::lock_guard lock(mutex);auto& e=entries[entry(id)];require(e.state!=JobState::FAILED&&e.state!=JobState::RETIRED,"terminal ticket");require(!e.ticket||e.ticket_identity==ticket_identity,"ticket conflict");e.ticket_identity=ticket_identity;e.ticket=true;if(e.state==JobState::ABSENT)e.state=JobState::QUEUED;cv.notify_all();}
 void publish(const SnapshotId& id,uint64_t fp,uint64_t boundary,uint64_t charge=1,bool coalesce=true){std::lock_guard lock(mutex);auto& e=entries[entry(id)];
  require(!e.fingerprint_set||(e.fingerprint==fp&&e.boundary==boundary),"input changed");if(e.payload)return;
  require(e.state==JobState::ABSENT||e.state==JobState::QUEUED,"invalid publication");
  if(coalesce)for(auto& old:entries)if(old.used&&old.id.same_namespace(id)&&old.id.end<id.end&&!old.refs&&old.state==JobState::QUEUED)old.payload=false;
  e.fingerprint_set=true;e.fingerprint=fp;e.boundary=boundary;
  for(;;){size_t pending=0;uint64_t bytes=0;ExactPureJob* oldest=nullptr;
   for(auto& old:entries)if(old.used&&old.payload){bytes+=old.charge;if(old.state==JobState::QUEUED||old.state==JobState::RUNNING)++pending;if(!old.refs&&old.priority>=2&&old.state==JobState::QUEUED&&(!oldest||old.order<oldest->order))oldest=&old;}
   if(pending<40&&charge<=64*1024*1024&&bytes<=64*1024*1024-charge)break;
   require(oldest,"input credit exhausted");retire(*oldest,Reason::BackgroundCredit);
  }
  e.charge=charge;e.payload=true;e.state=JobState::QUEUED;cv.notify_all();
 }
 void admit(const SnapshotId& id,uint64_t command,int64_t deadline){std::lock_guard lock(mutex);
  for(auto& c:commands)require(!c.used||c.id!=command,"duplicate command");
  auto c=std::find_if(commands.begin(),commands.end(),[](auto& x){return !x.used;});require(c!=commands.end(),"command capacity");
  auto i=entry(id);auto& e=entries[i];require(e.state!=JobState::FAILED&&e.state!=JobState::RETIRED,"terminal admission");
  *c={true,command,i,deadline};++e.refs;e.priority=e.state==JobState::RUNNING?1:0;cv.notify_all();
 }
 bool claim(uint64_t boundary,SnapshotId& out,bool background=true){std::lock_guard lock(mutex);size_t best=N;int64_t best_deadline=INT64_MAX;
  for(size_t i=0;i<N;++i){auto& e=entries[i];if(!e.used||e.state!=JobState::QUEUED||!e.payload)continue;
   bool latest=true;for(auto& n:entries)if(n.used&&n.id.same_namespace(e.id)&&n.id.end>e.id.end)latest=false;
   if(!e.refs&&(!background||!latest||e.boundary>boundary))continue;
   int64_t deadline=INT64_MAX;for(auto& c:commands)if(c.used&&c.job==i)deadline=std::min(deadline,c.deadline);
   if(best==N||std::tuple(!bool(e.refs),e.priority,e.refs?deadline:e.id.end,e.order)<std::tuple(!bool(entries[best].refs),entries[best].priority,entries[best].refs?best_deadline:entries[best].id.end,entries[best].order)){best=i;best_deadline=deadline;}
  }if(best==N)return false;auto& e=entries[best];e.state=JobState::RUNNING;++e.computes;out=e.id;return true;
 }
 void promote_exact(const SnapshotId& id,uint8_t priority){std::lock_guard lock(mutex);auto i=find(id);require(i<N&&priority<=2,"exact promotion state");entries[i].priority=std::min(entries[i].priority,priority);}
 // The external physical B20 scheduler selects the exact immutable slot.
 void start_exact(const SnapshotId& id){std::lock_guard lock(mutex);auto i=find(id);require(i<N&&entries[i].state==JobState::QUEUED&&entries[i].payload,"exact dispatch state");entries[i].state=JobState::RUNNING;++entries[i].computes;}
 void retire_exact(const SnapshotId& id){std::lock_guard lock(mutex);auto i=find(id);require(i<N,"missing retirement");auto& e=entries[i];retire(e,Reason::VersionLifetime);Floor* floor=nullptr;for(auto& f:floors)if(f.used&&f.id.same_namespace(id)){floor=&f;break;}if(!floor)for(auto& f:floors)if(!f.used){floor=&f;break;}require(floor,"retired namespace capacity");if(!floor->used||floor->id.end<id.end)*floor={true,id};e.used=false;}
 void complete(const SnapshotId& id,double value){std::lock_guard lock(mutex);auto i=find(id);require(i<N&&entries[i].state==JobState::RUNNING,"completion state");auto& e=entries[i];e.payload=false;e.result=value;e.state=JobState::READY;cv.notify_all();}
 void fail(const SnapshotId& id){std::lock_guard lock(mutex);auto i=find(id);require(i<N,"missing job");auto& e=entries[i];require(e.state!=JobState::READY&&e.state!=JobState::RETIRED,"failure state");e.payload=false;e.state=JobState::FAILED;e.reason=Reason::ModelFailure;cv.notify_all();}
 template<class Validate> double wait(const SnapshotId& id,Validate validate){for(;;){validate();std::unique_lock lock(mutex);auto i=find(id);require(i<N,"orphan future");auto& e=entries[i];if(e.state==JobState::READY)return e.result;require(e.state!=JobState::FAILED&&e.state!=JobState::RETIRED&&!closed,"terminal future");cv.wait_for(lock,std::chrono::milliseconds(50));}}
 void release(uint64_t command){std::lock_guard lock(mutex);for(auto& c:commands)if(c.used&&c.id==command){--entries[c.job].refs;c.used=false;}cv.notify_all();}
 void prune(size_t versions){std::lock_guard lock(mutex);versions_=versions;prune_locked(versions);}

 bool lookup(const SnapshotId& id,ExactPureJob& out){std::lock_guard lock(mutex);auto i=find(id);if(i==N)return false;out=entries[i];return true;}
 ExactPureJob inspect(const SnapshotId& id){std::lock_guard lock(mutex);auto i=find(id);require(i<N,"missing job");return entries[i];}
 void close(){std::lock_guard lock(mutex);closed=true;for(auto& e:entries)if(e.used&&(e.state==JobState::ABSENT||e.state==JobState::QUEUED)){e.payload=false;e.state=JobState::FAILED;e.reason=Reason::Shutdown;}cv.notify_all();}
};
// A command's lifetime pins BOTH the exact future and the immutable snapshot.
// The store and registry outlive these handles. Construct only after admission
// validation succeeds; the externally supplied command validator owns expiry.
template<class Store,class Registry> class AdmittedCommand {
 Registry& jobs; uint64_t command;
public:
 typename Store::Ref snapshot;
 AdmittedCommand(Store& store,Registry& registry,const SnapshotId& id,uint64_t cmd,int64_t deadline)
  :jobs(registry),command(cmd),snapshot(store.acquire(id)){jobs.admit(id,cmd,deadline);}
 AdmittedCommand(const AdmittedCommand&)=delete;
 AdmittedCommand& operator=(const AdmittedCommand&)=delete;
 ~AdmittedCommand(){jobs.release(command);}
};
}
