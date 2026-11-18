#pragma once
#include "source.hpp"
namespace shadow::minimal {
enum InputGate:uint32_t {NoObservation=1,SourceNotStarted=2,SourceNotFresh=4,HistoryShort=8,NonfiniteHistory=16,Expired=32,OldBar=64,OldSource=128,QueueFull=256};
inline uint32_t input_gate(bool seen,bool started,bool live,bool fresh,size_t labeled,size_t packed,size_t invalid,int64_t eligible,int64_t bar,int64_t event,int64_t feed,int64_t mono,int64_t wall,size_t queue_depth){
 uint32_t flags=0;if(!seen)flags|=NoObservation;if(!started)flags|=SourceNotStarted;if(live&&!fresh)flags|=SourceNotFresh;
 if(labeled!=192||packed!=184){flags|=HistoryShort;}if(invalid){flags|=NonfiniteHistory;}
 if(live&&seen){if(mono<eligible||mono-eligible>=NS)flags|=Expired;if(wall<bar||wall-bar>5*NS)flags|=OldBar;if(!stage1::fresh_time(wall,event,feed))flags|=OldSource;}
 if(queue_depth>=12){flags|=QueueFull;}return flags;
}
inline const char* input_reason(uint32_t f){for(auto v:{NoObservation,SourceNotStarted,SourceNotFresh,HistoryShort,NonfiniteHistory,Expired,OldBar,OldSource,QueueFull})if(f&v){switch(v){case NoObservation:return "NO_OBSERVATION";case SourceNotStarted:return "SOURCE_NOT_STARTED";case SourceNotFresh:return "SOURCE_NOT_FRESH";case HistoryShort:return "HISTORY_NOT_READY";case NonfiniteHistory:return "NONFINITE_HISTORY";case Expired:return "DEADLINE_EXPIRED";case OldBar:return "BAR_STALE_OR_FUTURE";case OldSource:return "SOURCE_EVENT_STALE_OR_FUTURE";case QueueFull:return "INFERENCE_QUEUE_FULL";}}return "READY";}
// Only the canonical consumer writes these scalars. Monitoring reads atomics;
// it never reads concurrently mutable history or SDK payload storage.
struct SymbolGateTelemetry {
 std::atomic<uint64_t> observations{0},labeled{0},packed{0},invalid{0},accepted{0},source_rejected{0},history_rejected{0},stale_rejected{0},queue_rejected{0},model_started{0},model_completed{0},expired_before_start{0},completed_late{0},completed_stale{0};
 std::atomic<int64_t> eligible{0},bar{0},event{0},feed{0};std::atomic<uint32_t> last_gate{NoObservation};
};
}
