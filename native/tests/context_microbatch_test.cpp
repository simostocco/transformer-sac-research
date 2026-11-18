#include "context_microbatch.hpp"
#include <iostream>
using namespace shadow; using namespace shadow::stage1;
void test_context_microbatch() {
 using namespace integrated;
 std::array<minimal::SymbolGateTelemetry,SmallCapSlots> gates;
 std::array<std::atomic<uint64_t>,SmallCapSlots> deferred{},cold{};
 std::atomic<uint64_t> total_deferred{0},total_cold{0};
 minimal::PortfolioCapture capture(2*NS);
 minimal::ContextMicrobatch scheduler(gates,deferred,cold,total_deferred,total_cold,&capture);
 const int64_t start=now(),bar=1794257790LL*NS;
 auto publish=[&](uint32_t slot,int64_t end,double price){
  Job job;
  job.id.symbol=slot;job.id.bar=end;job.id.generation=1;job.eligible=job.enqueued=start;
  LabeledRow row;row.bar.close=price;
  for(size_t i=0;i<192;++i){
   row.bar.ts=end-int64_t(191-i)*NS;job.history.model.labeled.push(row);
  }
  InputRow input;input.values.fill(float(price));input.k1=input.k2=0;input.log_close=float(std::log(price));
  for(size_t i=0;i<FlatPackedRows;++i)job.history.packed.push(input);
  scheduler.publish(job);
 };
 publish(0,bar-NS,4);publish(0,bar,4);publish(1,bar,8);
 for(int i=0;i<6;++i)publish(2,bar-(100-i)*NS,16);
 require(scheduler.depth==7,"bounded per-symbol pending lane");
 std::atomic<bool> fresh{true};
 int fd[2];require(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"microbatch test socket");
 std::vector<Outcome> outcomes;
 std::exception_ptr worker_error;
 std::jthread worker([&]{
  try{
   size_t sac_calls=0;
   while(true){
    Header request;transfer(fd[1],&request,sizeof(request),false);
    if(request.stage==2){char ack[]="STOP_ACK";transfer(fd[1],ack,8,true);break;}
    Reply response;response.sequence=request.sequence;response.identity=request.identity;
    if(request.stage==0){
     require(request.count>0&&request.count<=20,"microbatch TF batch bound");
     std::vector<pure::v2::PreparedInput> inputs(request.count);
     transfer(fd[1],inputs.data(),inputs.size()*sizeof(inputs[0]),false);
     std::vector<NativeFlatObservation::TF> tf(inputs.size());
     for(size_t i=0;i<tf.size();++i)tf[i].fill(inputs[i].x[0]);
     transfer(fd[1],&response,sizeof(response),true);transfer(fd[1],tf.data(),tf.size()*sizeof(tf[0]),true);
    }else{
     require(request.stage==1&&request.count>0&&request.count<=4,"microbatch SAC batch bound");
     std::vector<std::array<float,490>> observations(request.count);
     transfer(fd[1],observations.data(),observations.size()*sizeof(observations[0]),false);
     std::vector<float> actions(request.count);
     for(size_t i=0;i<actions.size();++i){
      require(std::abs(observations[i][0]-observations[i][99]*10'000)<1e-6f,"symbol TF and price match");
      actions[i]=observations[i][99]*100;
     }
     if(++sac_calls==3)fresh=false;
     transfer(fd[1],&response,sizeof(response),true);transfer(fd[1],actions.data(),actions.size()*sizeof(float),true);
    }
   }
  }catch(...){worker_error=std::current_exception();}
  close(fd[1]);
 });
 try{
  scheduler.run(fd[0],[&]{return minimal::ContextMicrobatch::Clock{now(),bar+2*NS,1,fresh};},
                [&]{return outcomes.size()==3||now()>start+2*NS;},
                [&](const Outcome& outcome){outcomes.push_back(outcome);},[](size_t){});
 }catch(...){
  close(fd[0]);worker.join();
  if(worker_error){try{std::rethrow_exception(worker_error);}catch(const std::exception& e){std::cerr<<"Mock worker: "<<e.what()<<'\n';}}
  throw;
 }
 close(fd[0]);worker.join();
 if(worker_error)std::rethrow_exception(worker_error);
 scheduler.close();
 require(outcomes.size()==3&&outcomes[0].id.bar==bar-NS&&outcomes[0].disposition==0&&
         outcomes[1].id.bar==bar&&outcomes[1].id.symbol==0&&outcomes[1].disposition==0&&
         outcomes[2].id.symbol==1&&outcomes[2].disposition==7,"ordered FIFO and post-SAC stale gate");
 require(capture.tf_contexts==131&&capture.tf_batches==7&&capture.inputs.size()==2&&total_cold==6&&
         scheduler.primed(0,1)&&!scheduler.primed(0,2),"context batching, drain and generation identity");
}
int main(){test_context_microbatch();std::cout<<"CONTEXT_MICROBATCH=PASS\n";}
