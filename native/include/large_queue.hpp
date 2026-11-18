#pragma once
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <filesystem>
#include <string>
namespace shadow::stage1 {
inline int64_t queue_now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline constexpr uint64_t GiB=1ULL<<30,QueueMemoryLimit=16*GiB,MemoryReserve=8*GiB;
inline uint64_t available_memory(){
 std::ifstream mem("/proc/meminfo");std::string line;uint64_t available=0;
 while(std::getline(mem,line))if(line.starts_with("MemAvailable:"))available=std::stoull(line.substr(13))*1024;
 require(available>0,"memory availability unavailable");
 std::ifstream groups("/proc/self/cgroup");while(std::getline(groups,line))if(line.starts_with("0::")){
  auto root=std::filesystem::path("/sys/fs/cgroup");auto p=root/line.substr(3).substr(1);
  for(;;){uint64_t used=0;std::ifstream current(p/"memory.current");if(std::filesystem::exists(p/"memory.current")){require(bool(current>>used),"cgroup memory usage unreadable");}if(current)for(auto name:{"memory.max","memory.high"}){std::ifstream f(p/name);std::string v;if(std::filesystem::exists(p/name)){require(bool(f>>v),"cgroup memory limit unreadable");}if(!v.empty()&&v!="max"){auto limit=std::stoull(v);available=std::min(available,uint64_t(limit>used?limit-used:0));}}if(p==root)break;p=p.parent_path();}
 }
 return available;
}
inline size_t source_queue_capacity(){auto* text=std::getenv("LIVE_SHADOW_SOURCE_CAPACITY");if(!text)return 1000000;std::string s(text);require(!s.empty()&&s.find_first_not_of("0123456789")==std::string::npos,"invalid source capacity");auto n=std::stoull(s);require(n>=1024,"source capacity minimum1024");return size_t(n);}
// Single producer/single consumer. All payload and timestamp storage is fully
// constructed before source startup. Borrowed slots remain occupied through fn.
template<class T>class LargeSourceQueue {
 const size_t count;std::unique_ptr<T[]> rows;std::unique_ptr<std::atomic<int64_t>[]> stamps;
 alignas(64)std::atomic<uint64_t> in{0};alignas(64)std::atomic<uint64_t> out{0};
 std::atomic<uint64_t> occupied{0},maximum{0},overflows{0};
 std::atomic<int64_t> maximum_wait{0},backlog_since{0},maximum_recovery{0};
 static void high(std::atomic<int64_t>& value,int64_t n){auto old=value.load(std::memory_order_relaxed);while(old<n&&!value.compare_exchange_weak(old,n,std::memory_order_relaxed)){};}
 void release(uint64_t o){auto prior=occupied.fetch_sub(1);out.store(o+1,std::memory_order_release);if(prior==1){auto since=backlog_since.exchange(0);if(since)high(maximum_recovery,queue_now()-since);}}
public:
 static constexpr size_t slot_bytes=sizeof(T)+sizeof(std::atomic<int64_t>);
 explicit LargeSourceQueue(size_t n=source_queue_capacity()):count(n){
  require(n>0&&n<=(QueueMemoryLimit-4096)/slot_bytes,"source queue exceeds16GiB hard allocation limit");
  require(available_memory()>bytes()+MemoryReserve+2*GiB,"insufficient memory for queue plus10GiB reserve");
  rows=std::make_unique<T[]>(count);stamps=std::make_unique<std::atomic<int64_t>[]>(count);
  // Explicit first touch, including timestamps, before a callback can execute.
  for(size_t i=0;i<count;++i)stamps[i].store(0,std::memory_order_relaxed);
 }
 size_t capacity()const{return count;}uint64_t bytes()const{return uint64_t(count)*slot_bytes+4096;}
 bool push(const T& value)noexcept{auto i=in.load(std::memory_order_relaxed);if(i-out.load(std::memory_order_acquire)==count){++overflows;return false;}auto stamp=queue_now();rows[i%count]=value;stamps[i%count].store(stamp,std::memory_order_release);auto depth=occupied.fetch_add(1)+1;if(depth>maximum.load())maximum=depth;if(depth>=1024){int64_t zero=0;backlog_since.compare_exchange_strong(zero,stamp);}in.store(i+1,std::memory_order_release);return true;}
 template<class F>bool consume(F&& fn){auto o=out.load(std::memory_order_relaxed);if(o==in.load(std::memory_order_acquire))return false;high(maximum_wait,queue_now()-stamps[o%count].load(std::memory_order_acquire));try{fn(static_cast<const T&>(rows[o%count]));}catch(...){release(o);throw;}release(o);return true;}
 bool pop(T& value){return consume([&](const T& row){value=row;});}
 uint64_t accepted_count()const{return in.load();}uint64_t dequeued_count()const{return out.load();}uint64_t depth()const{return occupied.load();}uint64_t high_water()const{return maximum.load();}uint64_t overflow_count()const{return overflows.load();}
 int64_t max_wait_ns()const{return maximum_wait.load();}int64_t max_recovery_ns()const{return maximum_recovery.load();}
 int64_t backlog_age_ns()const{auto since=backlog_since.load();return since?std::max<int64_t>(0,queue_now()-since):0;}
 // Monitor reads only atomic timestamps, never a mutable payload. Retry if the
 // consumer advanced while sampling. This is an observational snapshot.
 int64_t oldest_age_ns()const{for(int retry=0;retry<3;++retry){auto o=out.load(std::memory_order_acquire);if(o==in.load(std::memory_order_acquire))return 0;auto t=stamps[o%count].load(std::memory_order_acquire);if(o==out.load(std::memory_order_acquire))return t?std::max<int64_t>(0,queue_now()-t):0;}return 0;}
};
}
