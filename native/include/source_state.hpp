#pragma once
#include "core.hpp"
namespace shadow {
struct SourceStamp {uint64_t session=0,source_epoch=0,generation=0,execution_epoch=0,sequence=0,outer=0;int64_t feed=0;uint8_t mode=0;};
enum class ClosureKind:uint8_t {Watermark,Trade};
struct BarCoordinate {uint32_t symbol=0;int64_t ts=0;uint64_t source_sequence=0,finalization_sequence=0,session=0,source_epoch=0,generation=0,execution_epoch=0,outer=0;int64_t feed=0;ClosureKind kind=ClosureKind::Watermark;};
// Source-coordinate owner. Sparse replay deliveries preserve actual original
// sequence numbers; no sequential renumbering or provenance synthesis occurs.
template<size_t Symbols=320> struct ProvenanceCore {
 struct State {int64_t current=-1,last=-1;uint64_t first=0;};std::array<State,Symbols> symbols{};
 const uint64_t session,source_epoch;int64_t feed=0;uint64_t sequence=0;
 ProvenanceCore(uint64_t s,uint64_t e):session(s),source_epoch(e){}
 template<class Emit>void close(uint32_t symbol,const SourceStamp& stamp,ClosureKind kind,Emit emit){auto& s=symbols[symbol];emit(BarCoordinate{symbol,s.current,s.first?s.first:stamp.sequence,stamp.sequence,stamp.session,stamp.source_epoch,stamp.generation,stamp.execution_epoch,stamp.outer,stamp.feed,kind});s.last=s.current;s.current+=NS;s.first=0;}
 void advance_source(const SourceStamp& stamp){require(stamp.session==session&&stamp.source_epoch==source_epoch,"new source owner required");require(stamp.feed>=feed&&stamp.sequence>=sequence,"source coordinate regression");feed=stamp.feed;sequence=stamp.sequence;}
 template<class Emit,class Before=std::less<size_t>>void delivery(const SourceStamp& stamp,Emit emit,Before before={}){advance_source(stamp);
  for(;;){size_t best=Symbols;int64_t due=INT64_MAX;for(size_t i=0;i<Symbols;++i)if(symbols[i].current>=0&&symbols[i].current+6*NS<=feed&&(symbols[i].current+6*NS<due||(symbols[i].current+6*NS==due&&best!=Symbols&&before(i,best)))){best=i;due=symbols[i].current+6*NS;}if(best==Symbols)break;close(uint32_t(best),stamp,ClosureKind::Watermark,emit);}
 }
 template<class Emit>void contribution(uint32_t symbol,int64_t event,const SourceStamp& stamp,Emit emit){require(symbol<Symbols,"source symbol bound");auto& s=symbols[symbol];int64_t second=event/NS*NS;require(second>s.last,"late contribution in preserved accepted corpus");if(s.current<0)s.current=second;require(second>=s.current,"out of order contribution");while(s.current<second)close(symbol,stamp,ClosureKind::Trade,emit);if(!s.first)s.first=stamp.sequence;}
};
}
