#pragma once
#include "flat_history.hpp"
#include "source.hpp"
#include <poll.h>
#include <sys/socket.h>
namespace shadow::integrated {
inline constexpr size_t InferenceQueueCapacity=12;
inline constexpr uint64_t Magic=0x494e544750553031;
struct Header{uint64_t magic=Magic,sequence=0;uint32_t stage=0,count=0;uint64_t identity=0;};
struct Reply{uint64_t magic=Magic,sequence=0,identity=0,received=0,completed=0;double transfer_ms=0,compute_ms=0,materialize_ms=0;};
static_assert(sizeof(Header)==32&&sizeof(Reply)==64&&sizeof(pure::v2::PreparedInput)==17280);
inline int64_t now(){return stage1::monotonic_now_ns();}
inline void transfer(int fd,void* payload,size_t bytes,bool send){auto* p=static_cast<char*>(payload);while(bytes){pollfd item{fd,short(send?POLLOUT:POLLIN),0};int ready;do{ready=poll(&item,1,15000);}while(ready<0&&errno==EINTR);require(ready>0,"GPU worker I/O timeout");auto n=send?::send(fd,p,bytes,MSG_NOSIGNAL) : recv(fd,p,bytes,0);if(n<0&&errno==EINTR)continue;require(n>0,"GPU worker disconnected");bytes-=size_t(n);p+=n;}}
inline void validate(const Header& h,const Reply& r){require(r.magic==Magic&&r.sequence==h.sequence&&r.identity==h.identity,"GPU observation identity mismatch");require(r.completed>=r.received&&std::isfinite(r.compute_ms)&&r.compute_ms>=0&&std::isfinite(r.transfer_ms)&&r.transfer_ms>=0&&std::isfinite(r.materialize_ms)&&r.materialize_ms>=0,"GPU timing invalid");}
inline int64_t flat_target(float action,double price){require(std::isfinite(action)&&std::isfinite(price)&&price>0,"invalid diagnostic decision");auto maximum=int64_t(std::max(1.0,std::floor(10000/price)));auto delta=int64_t(std::nearbyint(double(std::clamp(action,-1.0f,1.0f))*double(maximum)));return std::clamp<int64_t>(delta,0,std::min<int64_t>(maximum,10000));}
struct Identity{uint64_t job=0,source=0,finalization=0,session=20261020,epoch=1,generation=0,execution_epoch=0,recorded_frontier=0;uint32_t symbol=0;int64_t bar=0;};
struct Job{bool warm=false;Identity id;int64_t eligible=0,canonical=0,features=0,enqueued=0,event_started=0;NativeFlatHistory history;};
struct Outcome{Identity id;int64_t eligible=0,canonical=0,features=0,enqueued=0,event_started=0,started=0,prepared=0,tf_completed=0,sac_prepared=0,sac_completed=0,completed=0;double source_age_ms=0,replay_data_age_ms=0,tf_transfer_ms=0,tf_ms=0,tf_materialize_ms=0,sac_transfer_ms=0,sac_ms=0;float action=0;int64_t target=0;uint32_t disposition=0,tf_computed=0;};
// Disposition0 is diagnostic-only complete;1 admission-full;2 expired-before-start;
// 3 complete-but-late;4 model error. There is no executable disposition.
inline bool expired(int64_t eligible,int64_t completed){require(completed>=eligible,"decision monotonic regression");return completed-eligible>=NS;}
}
