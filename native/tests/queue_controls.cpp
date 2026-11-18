#include "protocol.hpp"
#include <iostream>
using namespace shadow;using namespace shadow::stage1;
int main(){try{
 Spsc<int,2> q;require(q.push(11)&&q.push(22),"fill slots");
 require(q.consume([&](const int& value){require(value==11&&!q.push(33)&&q.depth()==2,"borrowed slot cannot be overwritten or released early");}),"consume view");
 require(q.push(33),"slot released after callback");bool caught=false;try{q.consume([](const int&){throw 1;});}catch(int){caught=true;}require(caught&&q.depth()==1&&q.dequeued_count()==2,"exception release exactly once");int last=0;require(q.pop(last)&&last==33,"FIFO preserved");
 using O=Owner<1024,64>;auto a=std::make_unique<O>(std::make_unique<Simulator>(std::span<const Event>{}),1,1,ObservationConfiguration{},std::vector<std::string>{},false);auto b=std::make_unique<O>(std::make_unique<Simulator>(std::span<const Event>{}),1,1,ObservationConfiguration{},std::vector<std::string>{},false);
 auto copy=std::make_unique<StateEngine<2,O>>(*a),view=std::make_unique<StateEngine<2,O>>(*b);uint64_t closures_a=0,closures_b=0;
 auto send=[&](Event e){require(a->callback(e)==b->callback(e),"equal admission result");
  bool x=copy->step([&](auto,auto,const auto&){++closures_a;});bool y=view->step_in_place([&](auto,auto,const auto&){++closures_b;});require(x==y,"every admitted input consumed exactly once");
  require(copy->canonical.last_feed==view->canonical.last_feed&&copy->provenance.feed==view->provenance.feed&&copy->provenance.sequence==view->provenance.sequence,"every clock and source sequence preserved");
  for(size_t s=0;s<2;++s){require(copy->canonical.symbols[s].current==view->canonical.symbols[s].current&&copy->provenance.symbols[s].first==view->provenance.symbols[s].first&&copy->features[s].warm.size==view->features[s].warm.size,"canonical/provenance parity");if(copy->features[s].warm.size)require(!std::memcmp(copy->features[s].warm.back().values.data(),view->features[s].warm.back().values.data(),sizeof(FeatureRow)),"bitwise N1 parity");}
 };
 Event e;e.status=7;e.sdk.available=true;e.callback_kind=CallbackKind::STATUS;e.stamp.feed=e.wall_ns=NS;send(e);e.status=0;
 for(int i=1;i<=120000;++i){e.stamp.feed=e.wall_ns=NS+int64_t(i)*1000000;e.event_valid=e.accepted=false;e.callback_kind=CallbackKind::STATUS;send(e);if(i%17==0)send(e);
  if(i%250==0){e.callback_kind=CallbackKind::QUOTE;send(e);e.callback_kind=CallbackKind::TRADE;e.accepted=e.event_valid=true;e.event_ns=e.stamp.feed;e.symbol=uint32_t(i/250%2);e.price=5+e.symbol;e.size=10;send(e);e.callback_kind=CallbackKind::STATUS;e.event_valid=e.accepted=false;send(e);}
 }
 auto ah=a->health(0),bh=b->health(0);require(ah.callbacks==bh.callbacks&&ah.canonical_admitted==bh.canonical_admitted&&ah.canonical_dequeued==bh.canonical_dequeued&&ah.status_noop_skipped==bh.status_noop_skipped&&ah.overflow==0&&bh.overflow==0&&closures_a==closures_b&&closures_a>0,"lossless callback/admission/closure parity");
 std::cout<<"BORROW_LIFETIME_AND_EXCEPTION_RELEASE=PASS\nBITWISE_CANONICAL_N1_PARITY=PASS\nCLOCK_ADVANCES=120000\nCALLBACKS="<<ah.callbacks<<"\nADMITTED="<<ah.canonical_admitted<<"\nDEQUEUED="<<bh.canonical_dequeued<<"\nIDEMPOTENT_SKIPS="<<ah.status_noop_skipped<<"\nCLOSURES="<<closures_a<<"\nEVENT_LOSS=0\nREAL_SOURCE_STARTED=NO\nPROCESSTAPE_CALLED=NO\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
