#include "version_store.hpp"
#include <atomic>
#include <iostream>
#include <thread>
using namespace shadow;
SnapshotId id(int64_t end=10){return {20261027,1,7,1,1,1,0,end};}
template<class F> void rejects(F f){bool caught=false;try{f();}catch(const std::logic_error&){caught=true;}require(caught,"expected rejection");}
int main(){try{
 {SymbolState s;Ring<CanonicalBar,5000> bars;auto emit=[&](auto b){bars.push(b);};s.trade(100*NS,2.7,3,'B',emit);s.tick(106*NS-1,emit);require(!bars.size,"early close");s.tick(4105*NS,emit);require(bars.size==4000,"every intermediate second");require(bars.back().forward&&bars.back().close==2.7&&!bars.back().volume,"empty semantics");s.trade(100*NS,3,2,'A',emit);require(s.late==1,"late");rejects([&]{s.tick(0,emit);});}
 {CanonicalCore<2> source(20261027,1);Ring<int,10> order;
  source.symbols[0].trade(100*NS,2.7,3,'B',[](auto){});source.symbols[1].trade(100*NS,3,2,'A',[](auto){});
  source.tick(20261027,1,107*NS,[&](size_t i,auto){order.push(int(i));});
  require(order.size==4&&order.back(3)==0&&order.back(2)==1&&order.back(1)==0&&order.back()==1,"canonical tie order");
  require(source.history_versions[0]==1&&source.history_versions[1]==1,"one version per changed tick");
  rejects([&]{source.tick(20261027,2,110*NS,[](auto,auto){});});}
 {static VersionStore<1,2,0,0> pool;ExecutionPackage p;p.id=id();p.bar.close=1;require(pool.publish(p),"publish");auto old=pool.acquire_latest(0);p.id=id(11);p.bar.close=2;require(pool.publish(p),"supersede");auto newer=pool.acquire_latest(0);p.id=id(12);require(pool.publish(p),"third");p.id=id(13);require(!pool.publish(p),"bounded pins");require(old.get().bar.close==1&&newer.get().bar.close==2,"immutable old");}
 {static VersionStore<1,2,0,0> pool;ExecutionPackage p;p.id=id();p.bar.close=10;pool.publish(p);std::atomic<bool> done=false;std::thread reader([&]{while(!done){auto r=pool.acquire_latest(0);require(r.get().bar.close==double(r.get().id.end),"partial version");}});
  for(int i=11;i<1000;++i){p.id=id(i);p.bar.close=i;while(!pool.publish(p)){} }
  done=true;reader.join();}
 {Jobs<> jobs;jobs.ticket(id());jobs.admit(id(),1,500);SnapshotId out;require(!jobs.claim(100,out),"ticket not materialized");jobs.publish(id(),9,10);jobs.admit(id(),2,501);require(jobs.claim(100,out)&&out==id(),"claim");require(!jobs.claim(100,out),"one compute");std::array<double,3> values{};std::array<std::thread,3> readers;
  for(size_t i=0;i<3;++i){readers[i]=std::thread([&,i]{values[i]=jobs.wait(id(),[]{});});}
  jobs.complete(id(),12);for(auto& t:readers)t.join();for(auto v:values)require(v==12,"waiter");
  jobs.publish(id(11),10,11);jobs.prune(1);require(jobs.inspect(id()).computes==1,"pinned old");jobs.release(1);jobs.release(2);jobs.prune(1);rejects([&]{jobs.publish(id(),9,10);});}
 {Jobs<> j;SnapshotId out;j.publish(id(),9,10);j.publish(id(11),10,11);require(!j.inspect(id()).payload,"coalesce");j.admit(id(),1,10);j.publish(id(),9,10);require(j.claim(100,out)&&out==id(),"demand promotion");rejects([&]{j.wait(id(),[]{throw std::logic_error("expired");});});j.release(1);require(j.inspect(id()).state==JobState::RUNNING,"expiry leaves compute");j.complete(id(),1);rejects([&]{j.publish(id(11),99,11);});}
 {Jobs<> j;j.admit(id(),1,10);std::atomic<bool> woke=false;std::thread t([&]{rejects([&]{j.wait(id(),[]{});});woke=true;});j.close();t.join();require(woke,"shutdown wake");}
 {Jobs<> j;SnapshotId out;j.publish(id(),1,1);j.claim(1,out);std::thread t([&]{rejects([&]{j.wait(id(),[]{});});});j.fail(id());t.join();require(j.inspect(id()).state==JobState::FAILED,"failed");}
 {Ring<int,2> q;require(q.try_push(1)&&q.try_push(2)&&!q.try_push(3),"queue bound");int x;require(q.pop(x)&&x==1&&q.pop(x)&&x==2&&!q.pop(x),"queue order");}
 std::cout<<"PASS 10 native core scenarios\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
