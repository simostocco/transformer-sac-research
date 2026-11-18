#pragma once
#include "core.hpp"
namespace shadow {
enum class RiskReason:uint8_t {MaxDelta,MaxSymbol,Deadband,MaxGross,Cash,NotReady,ManualHalt,DailyLoss,Cutoff,Trailing,Authorization};
struct RiskReasons {uint64_t bits=0;void add(RiskReason r){bits|=uint64_t(1)<<unsigned(r);}bool has(RiskReason r)const{return bits&(uint64_t(1)<<unsigned(r));}};
struct PaperConfig {
 double initial=100000,max_symbol=10000,max_gross=100000,deadband=0,fee=.01,slippage=5;
 int64_t max_delta=5000;bool strict_cap=false,trailing=false,cutoff=false,manual_enabled=true,flatten_manual=false;
 double trailing_trigger=.5,trailing_drawdown=.15,daily_loss=0;int cutoff_minute=660;
};
struct PaperPosition {int64_t shares=0;double average=0,mark=0,peak=0;bool trailing_locked=false;};
struct PaperPending {bool used=false,restored=false;uint64_t intent=0;int64_t target=0,remaining=0;double created=0,eligible=0,updated=0,reference=0,reserved_cash=0,reserved_notional=0,action=0;int64_t raw_delta=0,raw_target=0,model_max_shares=0,model_max_delta=0;double timeout=0;uint64_t insertion=0;};
struct PaperMetrics {double initial=0,cash=0,available=0,reserved=0,reserved_notional=0,gross=0,equity=0,realized=0,unrealized=0,session_equity=0,daily=0;bool loss=false,halt=false;};
enum class FillStatus:uint8_t {NoPending,NotReady,RiskCancelled,Timeout,NoRealTrade,Partial,Full};
struct PaperOutcome {int64_t approved=0,filled=0,target=0;uint64_t intent=0;FillStatus status=FillStatus::NoPending;double fill_price=0,fees=0,slippage=0;RiskReasons reasons{};};
template<size_t Symbols=320> class PaperLedger {
 std::array<bool,Symbols> used{};std::array<uint32_t,Symbols> order{};size_t position_count=0;uint64_t pending_clock=0;
 std::array<uint32_t,Symbols> pending_order()const{std::array<uint32_t,Symbols> indices{};for(size_t i=0;i<Symbols;++i)indices[i]=uint32_t(i);std::sort(indices.begin(),indices.end(),[&](auto a,auto b){return pending[a].insertion<pending[b].insertion;});return indices;}
 PaperPosition& position(uint32_t s){require(s<Symbols,"paper symbol bound");if(!used[s]){used[s]=true;order[position_count++]=s;}return positions[s];}
 static int64_t affordable(double money,double unit){if(money<=0)return 0;return std::max<int64_t>(0,int64_t(std::floor((money+std::min(1e-8,unit*1e-9))/unit)));}
 double buy(double price)const{return price*(1+config.slippage/10000.0);}
 bool after_cutoff()const{return config.cutoff&&ny_minute>=config.cutoff_minute;}
 RiskReasons increase_blocks()const{RiskReasons r;if(manual_halt)r.add(RiskReason::ManualHalt);if(daily_loss)r.add(RiskReason::DailyLoss);if(after_cutoff())r.add(RiskReason::Cutoff);return r;}
 void check_loss(){if(config.daily_loss>0&&!daily_loss&&metrics().daily<=-config.daily_loss){daily_loss=true;for(auto& p:pending)if(p.used&&p.remaining>0)p.used=false;}}
 int64_t override_delta(uint32_t symbol,int64_t delta,int64_t model_max_shares,int64_t model_max_delta,RiskReasons& reasons){auto& p=position(symbol);
  if(config.trailing&&p.shares){double threshold=config.trailing_trigger*config.max_symbol,stop=p.peak*(1-config.trailing_drawdown),open=p.shares*(p.mark-p.average);if(p.peak>=threshold&&open<=stop)p.trailing_locked=true;}
  if(after_cutoff())reasons.add(RiskReason::Cutoff);else if(p.trailing_locked)reasons.add(RiskReason::Trailing);else if(manual_halt&&config.flatten_manual)reasons.add(RiskReason::ManualHalt);else return delta;
  return -std::min({p.shares,model_max_shares,model_max_delta});
 }
 int64_t limits(uint32_t symbol,int64_t delta,double price,RiskReasons& reasons,bool deadband=true){auto& p=position(symbol);int64_t clipped=std::clamp(delta,-config.max_delta,config.max_delta);if(clipped!=delta)reasons.add(RiskReason::MaxDelta);delta=std::max(-p.shares,clipped);
  int64_t cap=int64_t(std::floor(config.max_symbol/buy(price)));int64_t band=int64_t(std::nearbyint(config.deadband*std::max(1.0,std::floor(config.max_symbol/price))));
  auto apply_band=[&]{if(deadband&&config.deadband>0&&std::abs(delta)<=band){if(delta)reasons.add(RiskReason::Deadband);delta=0;}};
  if(config.strict_cap)apply_band();
  if(config.strict_cap&&p.shares+delta>cap){delta=cap-p.shares;reasons.add(RiskReason::MaxSymbol);}else if(delta>0&&delta>std::max<int64_t>(0,cap-p.shares)){delta=std::max<int64_t>(0,cap-p.shares);reasons.add(RiskReason::MaxSymbol);}
  if(!config.strict_cap){apply_band();}
  if(std::abs(delta)>config.max_delta){delta=std::clamp(delta,-config.max_delta,config.max_delta);reasons.add(RiskReason::MaxDelta);}
  if(delta>0){double other=0;for(auto s:pending_order())if(pending[s].used){auto& q=pending[s];other+=std::max(q.reserved_notional,std::max<int64_t>(0,q.remaining)*buy(std::max(positions[s].mark,q.reference)));}
   double gross=metrics().gross;gross+=p.shares*(buy(price)-p.mark);int64_t room=std::max<int64_t>(0,int64_t(std::floor((config.max_gross-gross-other)/buy(price))));if(delta>room){delta=room;reasons.add(RiskReason::MaxGross);}
   double available=cash-metrics().reserved;int64_t shares=affordable(available,buy(price)+config.fee);if(delta>shares){delta=shares;reasons.add(RiskReason::Cash);}}
  return delta;
 }
public:
 PaperConfig config;std::array<PaperPosition,Symbols> positions{};std::array<PaperPending,Symbols> pending{};
 double cash=0,realized=0,fees=0,slippage=0,session_equity=0,last_timestamp=0;uint64_t sequence=0;uint32_t session=0;int ny_minute=0;bool daily_loss=false,manual_halt=false;
 explicit PaperLedger(PaperConfig c={}):config(c),cash(c.initial),session_equity(c.initial){require(c.initial>0&&c.max_delta>0&&c.max_symbol>0&&c.max_gross>0&&c.fee>=0&&c.slippage>=0&&c.slippage<10000,"paper config");}
 PaperMetrics metrics()const{PaperMetrics m;m.initial=config.initial;m.cash=cash;for(size_t i=0;i<position_count;++i){auto& p=positions[order[i]];m.gross+=p.shares*p.mark;m.unrealized+=p.shares*(p.mark-p.average);}for(auto s:pending_order())if(pending[s].used){m.reserved+=pending[s].reserved_cash;m.reserved_notional+=pending[s].reserved_notional;}
  m.equity=cash+m.gross;m.available=cash-m.reserved;if(m.available>=-1e-8&&m.available<0)m.available=0;m.realized=realized;m.session_equity=session_equity;m.daily=m.equity-session_equity;m.loss=daily_loss;m.halt=manual_halt;return m;
 }
 void mark(uint32_t symbol,double timestamp,double price,uint32_t date,int minute){require(timestamp>0&&std::isfinite(timestamp)&&price>0&&std::isfinite(price),"paper mark numeric");require(timestamp>=last_timestamp,"out-of-order paper epoch");last_timestamp=timestamp;ny_minute=minute;auto& p=position(symbol);p.mark=price;
  if(date!=session){if(session)for(auto& q:pending)q.used=false;session=date;session_equity=metrics().equity;daily_loss=false;}
  if(p.shares){p.peak=std::max(p.peak,p.shares*(p.mark-p.average));}
  check_loss();RiskReasons reasons;override_delta(symbol,0,1,1,reasons);
 }
 void cancel(uint32_t symbol){require(symbol<Symbols,"paper symbol bound");pending[symbol].used=false;}
 PaperOutcome submit(uint32_t symbol,double timestamp,double price,double action,int64_t raw_delta,int64_t max_shares,int64_t max_delta,double timeout,double seconds,uint32_t date,int minute,bool ready=true,uint64_t blocks=0){require(std::isfinite(action)&&action>=-1&&action<=1,"raw action");mark(symbol,timestamp,price,date,minute);auto& p=position(symbol);auto old=pending[symbol];cancel(symbol);PaperOutcome out;
  auto requested=override_delta(symbol,raw_delta,max_shares,max_delta,out.reasons);auto delta=ready?limits(symbol,requested,price,out.reasons):0;
  if(delta>0&&blocks){out.reasons.bits|=blocks;delta=0;}if(delta>0&&increase_blocks().bits){out.reasons.bits|=increase_blocks().bits;delta=0;}if(!ready)out.reasons.add(RiskReason::NotReady);
  ++sequence;int64_t target=p.shares+delta;
  if(delta){bool same=!old.restored&&old.used&&old.target==target&&old.remaining==delta;pending[symbol]={true,false,sequence,target,delta,same?old.created:timestamp,same?old.eligible:timestamp+seconds,timestamp,price,std::max<int64_t>(0,delta)*(buy(price)+config.fee),std::max<int64_t>(0,delta)*buy(price),action,raw_delta,p.shares+raw_delta,max_shares,max_delta,timeout,++pending_clock};}
  out.approved=delta;out.target=target;out.intent=sequence;return out;
 }
 PaperOutcome fill(uint32_t symbol,double timestamp,double price,double real_trades,double volume,double participation,uint32_t date,int minute,bool execution_ready=true,uint64_t blocks=0){mark(symbol,timestamp,price,date,minute);auto& pos=position(symbol);auto old=pending[symbol];PaperOutcome out;if(!old.used)return out;out.intent=old.intent;out.target=old.target;
  double age=std::max(0.0,timestamp-old.created);if(!execution_ready||old.restored){out.status=FillStatus::NotReady;return out;}if(old.remaining>0&&blocks){cancel(symbol);out.status=FillStatus::RiskCancelled;out.reasons.bits=blocks;return out;}
  if(age>=old.timeout){cancel(symbol);out.status=FillStatus::Timeout;return out;}if(timestamp<old.eligible||!std::isfinite(real_trades)||real_trades<=0){out.status=FillStatus::NoRealTrade;return out;}
  cancel(symbol);int64_t remaining=limits(symbol,old.remaining,price,out.reasons,false);
  if(old.remaining>0){out.reasons.bits|=increase_blocks().bits;if(pos.trailing_locked)out.reasons.add(RiskReason::Trailing);if(out.reasons.has(RiskReason::ManualHalt)||out.reasons.has(RiskReason::DailyLoss)||out.reasons.has(RiskReason::Cutoff)||out.reasons.has(RiskReason::Trailing))remaining=0;remaining=std::max<int64_t>(0,remaining);}else remaining=std::max(old.remaining,std::min<int64_t>(0,remaining));
  if(!remaining){out.status=FillStatus::RiskCancelled;return out;}
  int64_t max_fill=std::isfinite(volume)&&volume>0?std::max<int64_t>(1,int64_t(std::floor(volume*std::max(0.0,participation)))):std::abs(remaining);
  int64_t quantity=std::min({std::abs(remaining),max_fill,config.max_delta,old.model_max_delta});
  if(remaining<0){double net=config.fee-price*(1-config.slippage/10000.0);if(net>0){auto shares=affordable(cash-metrics().reserved,net);if(quantity>shares){quantity=shares;out.reasons.add(RiskReason::Cash);}}}
  if(!quantity){out.status=FillStatus::RiskCancelled;return out;}
  int64_t delta=remaining>0?quantity:-std::min(quantity,pos.shares);double exec=price*(1+(config.slippage/10000.0)*(delta>0?1:-1)),fee=std::abs(delta)*config.fee,slip=std::abs(delta)*std::abs(exec-price);
  if(delta>0){pos.average=(pos.shares*pos.average+delta*price)/(pos.shares+delta);realized-=fee+slip;}else realized+=std::abs(delta)*(price-pos.average)-fee-slip;
  cash-=delta*exec+fee;fees+=fee;slippage+=slip;pos.shares+=delta;if(!pos.shares){pos.average=0;pos.peak=0;pos.trailing_locked=false;}remaining-=delta;
  if(remaining){old.remaining=remaining;old.target=pos.shares+remaining;old.updated=timestamp;old.reference=price;old.reserved_cash=std::max<int64_t>(0,remaining)*(buy(price)+config.fee);old.reserved_notional=std::max<int64_t>(0,remaining)*buy(price);old.used=true;old.insertion=++pending_clock;pending[symbol]=old;}
  check_loss();require(metrics().available>=-1e-8&&cash>=-1e-8&&pos.shares>=0,"paper economic invariant");if(cash<0)cash=0;
  out.filled=delta;out.approved=delta;out.target=pos.shares+(pending[symbol].used?pending[symbol].remaining:0);out.status=pending[symbol].used?FillStatus::Partial:FillStatus::Full;out.fill_price=exec;out.fees=fee;out.slippage=slip;return out;
 }
 void halt(bool resume=false){if(!config.manual_enabled)return;if(resume){manual_halt=false;return;}if(!manual_halt){manual_halt=true;for(auto& p:pending)if(p.used&&p.remaining>0)p.used=false;}}
};
using PaperState=PaperLedger<320>;
}
