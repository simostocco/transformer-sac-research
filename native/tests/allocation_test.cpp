#include "model_state.hpp"
#include "execution.hpp"
#include "version_store.hpp"
#include "inference_interface.hpp"
#include "pure_engine.hpp"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include "allocation_counter.hpp"
using namespace shadow;
template<class F> void audit(const char* name,F f){auto cpp=allocations.load(),c=c_allocations.load();f();cpp=allocations.load()-cpp;c=c_allocations.load()-c;require(cpp==0&&c==0,"hot path allocated");std::cout<<name<<" CPP_NEW="<<cpp<<" C_MALLOC_CALLOC_REALLOC="<<c<<'\n';}
int main(){
 static FeatureState features;features.config={10,10,10,180};static ModelFacingState model;CanonicalBar bar;bar.close=2.7;bar.volume=3;bar.trades=1;
 ClusterAssets main,ret;main.count=4;ret.count=3;main.std.fill(1);ret.std.fill(1);InputAssets assets;assets.std.fill(1);
 audit("BAR_1000",[&]{for(int i=0;i<1000;++i){bar.ts=int64_t(i)*NS;auto row=features.append(bar,i);model.append(bar,row,main,ret,assets);if(model.inputs.size==120){auto c=model.context();require(c.k1.size()==120,"context");}}});
 SymbolState canonical;canonical.trade(100*NS,2.7,1,'B',[](auto){});
 audit("CANONICAL_EMPTY_BAR_1000",[&]{canonical.tick(1106*NS,[&](auto b){features.append(b,0);});});
 static VersionStore<20> snapshots;static OwnedJobs<20,256> jobs(snapshots);std::mutex mutex;TokenAuthority<> authority;NativeOrderedPaper<20> paper(mutex,authority);
 ExecutionPackage package;package.bar=bar;package.history=features.warm;package.model_inputs=model.ready;package.id={20261027,1,437858,1,1,1,0,100*NS};
 audit("SNAPSHOT_PUBLICATION_1000",[&]{for(int i=0;i<1000;++i){package.id.end=(100+i)*NS;require(snapshots.publish(package),"snapshot capacity");}});
 audit("VALID_COMMAND_COMMIT_1000",[&]{for(int i=0;i<1000;++i){int64_t now=(10000+2*i)*NS;package.id.end=now-6*NS;require(snapshots.publish(package),"publish");
  AdmittedToken token;token.snapshot=package.id;token.command=uint64_t(i+1);token.source_generation=1;token.active_admission=10;token.source_sequence=token.finalization_sequence=uint64_t(i+1);token.admission_ns=token.event_ns=token.feed_ns=now;token.deadline_ns=now+5*NS;token=authority.issue(token);
  {AdmittedCommand command(snapshots,jobs.registry,package.id,token.command,token.deadline_ns);jobs.publish(package.id,1,token.source_sequence);SnapshotId claimed;require(jobs.registry.claim(token.source_sequence,claimed),"claim");jobs.complete(claimed,.1);require(jobs.registry.wait(claimed,[] {})==.1,"exact result");
   CurrentAuthority current{20261027,1,437858,1,1,10,true,true,true,true,true,true,true,true,now};current.frontier=current.model=1;double action=i%2?-.1:.1;
   require(authority.validate(token,current,now)==TokenOutcome::Current,"token");require(paper.commit(token,current,now,2.7,action,1000,5000,900,1,double(now)/NS,20261027,600)==TokenOutcome::Current,"commit");
   paper.bar(0,437858,1,package.id.end+NS,now+NS,double(now)/NS+1,2.7,1,1000,1,20261027,600,true);
  }
  authority.release(token.command);jobs.prune(1);
 }});
 static NativePureEngine<40,20> pure_jobs;
 struct Mock:ModelBackend {bool transformer(const PureInferenceRequest&,std::array<float,439>& out)override{out.fill(.1f);return true;}bool observation(const PureInferenceRequest&,const std::array<float,439>&,pure::v2::FlatObservation&)override{return false;}bool sac(const pure::v2::FlatObservation&,std::array<float,1>&,std::array<float,17>&)override{return false;}} mock;
 audit("PURE_JOB_MOCK_1000",[&]{for(int i=0;i<1000;++i){PureInferenceRequest q;q.key[0]=1;q.input_digest[0]=1;q.revisions.transformer=pure::v2::TransformerRevision;q.snapshot={20261027,1,7,1,1,1,0,(20000+i)*NS};q.source_cut={1,1,1,1};SnapshotId id{20261027,1,7,1,1,1,0,q.snapshot.end_ns};require(pure_jobs.publish(q),"pure publish");pure_jobs.admit(q,i+1,0);require(pure_jobs.service(mock,[](auto,const auto&){}),"pure service");require(pure_jobs.result(id),"pure ready");pure_jobs.release(i+1);pure_jobs.retire(id);}});
 std::cout<<"HEAP_ALLOCATIONS_PER_PURE_JOB=0\n";
 std::cout<<"HEAP_ALLOCATIONS_PER_BAR=0\nHEAP_ALLOCATIONS_PER_COMMAND=0\n";
}
