#pragma once
#include "core.hpp"
#include <optional>
namespace shadow {
template<size_t Symbols,size_t Versions=2,size_t CommandPins=128,size_t BackgroundPins=40> class VersionStore {
public:static constexpr size_t Capacity=Symbols*Versions+CommandPins+BackgroundPins+1;
private:
 struct Slot{ExecutionPackage package{};size_t refs=0;};std::array<Slot,Capacity> slots{};std::array<Ring<size_t,Versions>,Symbols> retained{};std::mutex mutex;
 void release(size_t index){std::lock_guard lock(mutex);require(slots[index].refs>0,"snapshot ref underflow");--slots[index].refs;}
public:
 class Ref {VersionStore* owner=nullptr;size_t index=0;friend class VersionStore;Ref(VersionStore* o,size_t i):owner(o),index(i){}
 public:Ref()=default;Ref(const Ref&)=delete;Ref& operator=(const Ref&)=delete;Ref(Ref&& r)noexcept:owner(std::exchange(r.owner,nullptr)),index(r.index){}
  Ref& operator=(Ref&& r)noexcept{if(this!=&r){if(owner)owner->release(index);owner=std::exchange(r.owner,nullptr);index=r.index;}return *this;}
  ~Ref(){if(owner)owner->release(index);}const ExecutionPackage& get()const{require(owner,"empty snapshot pin");return owner->slots[index].package;}
 };
 bool publish(const ExecutionPackage& p){std::lock_guard lock(mutex);require(p.id.symbol<Symbols,"snapshot symbol bound");for(auto& s:slots)require(!s.refs||!(s.package.id==p.id),"duplicate immutable snapshot identity");
  for(size_t i=0;i<Capacity;++i)if(!slots[i].refs){slots[i].package=p;slots[i].refs=1;auto& history=retained[p.id.symbol];if(history.size==Versions){size_t old;history.pop(old);--slots[old].refs;}history.push(i);return true;}
  return false;}
 Ref acquire_latest(uint32_t symbol){std::lock_guard lock(mutex);require(symbol<Symbols&&retained[symbol].size>0,"no ready snapshot");size_t i=retained[symbol].back();++slots[i].refs;return Ref(this,i);}
 Ref acquire(const SnapshotId& id){std::lock_guard lock(mutex);for(size_t i=0;i<Capacity;++i)if(slots[i].refs&&slots[i].package.id==id){++slots[i].refs;return Ref(this,i);}throw std::logic_error("exact snapshot not retained");}
};
// Payload pins are owned by the exact job until coalescence/completion/failure.
// Command pins are independently held by AdmittedCommand, so neither supersession
// nor payload release can invalidate an admitted command's immutable cut.
template<size_t Symbols,size_t Entries=4008> class OwnedJobs {
 using Store=VersionStore<Symbols>;struct Pin{bool used=false;SnapshotId id{};std::optional<typename Store::Ref> value;};std::array<Pin,Entries> pins{};Store& store;
 void collect(){for(auto& p:pins)if(p.used){ExactPureJob e;if(!registry.lookup(p.id,e)||!e.payload){p.value.reset();p.used=false;}}}
public:Jobs<Entries,Symbols> registry;
 explicit OwnedJobs(Store& s):store(s){}
 void publish(const SnapshotId& id,uint64_t fingerprint,uint64_t boundary){for(auto& p:pins)if(p.used&&p.id==id){registry.publish(id,fingerprint,boundary);return;}
  auto it=std::find_if(pins.begin(),pins.end(),[](const auto& p){return !p.used;});require(it!=pins.end(),"job pin capacity");auto pin=store.acquire(id);registry.publish(id,fingerprint,boundary);it->id=id;it->value.emplace(std::move(pin));it->used=true;collect();}
 const ExecutionPackage& payload(const SnapshotId& id){for(auto& p:pins)if(p.used&&p.id==id)return p.value->get();throw std::logic_error("job input not owned");}
 void complete(const SnapshotId& id,double result){registry.complete(id,result);collect();}
 void fail(const SnapshotId& id){registry.fail(id);collect();}
 void prune(size_t versions){registry.prune(versions);collect();}
 void close(){registry.close();collect();}
};
}
