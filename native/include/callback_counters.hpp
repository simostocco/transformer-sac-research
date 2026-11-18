#pragma once
#include <array>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace shadow::stage1 {
enum class CallbackKind:uint8_t {STATUS,TRADE,QUOTE,OTHER,UNKNOWN};
inline constexpr size_t CallbackCategoryCount=5;
struct CallbackCounts {
 std::array<uint64_t,CallbackCategoryCount> values{};
 uint64_t total()const noexcept{uint64_t n=0;for(auto v:values)n+=v;return n;}
};
// One writer: the existing serialized source callback. Readers are passive.
// Each independent cumulative bucket is exact and monotonic. The sample total
// is derived from the SAME bucket reads, so every sample reconciles, even while
// the source advances. This is not a transactional snapshot of source state.
// No reset method: lifetime is precisely the source Owner/attempt lifetime.
class CallbackCounters {
 static_assert(std::atomic<uint64_t>::is_always_lock_free);
 std::array<std::atomic<uint64_t>,CallbackCategoryCount> buckets{};
 uint64_t written=0; // single writer only; never read by the monitor
public:
 explicit CallbackCounters(CallbackCounts initial={}){
  for(size_t i=0;i<CallbackCategoryCount;++i){auto v=initial.values[i];if(v>UINT64_MAX-written)throw std::overflow_error("callback counter initial sum");written+=v;buckets[i].store(v,std::memory_order_relaxed);}
 }
 bool record(CallbackKind kind)noexcept{
  if(written==UINT64_MAX)return false; // caller fail-stops rather than wrap
  auto i=size_t(kind);if(i>=CallbackCategoryCount)i=size_t(CallbackKind::UNKNOWN);
  auto& bucket=buckets[i];bucket.store(bucket.load(std::memory_order_relaxed)+1,std::memory_order_release);++written;return true;
 }
 CallbackCounts snapshot()const noexcept{CallbackCounts c;for(size_t i=0;i<CallbackCategoryCount;++i)c.values[i]=buckets[i].load(std::memory_order_acquire);return c;}
};
}
