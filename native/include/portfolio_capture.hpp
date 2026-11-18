#pragma once

#include "protocol.hpp"
#include <vector>

namespace shadow::minimal {

// Bounded in-memory reference capture. Each vector has one producer.
struct PortfolioCapture {
 struct Bar { uint32_t slot; int64_t start, feed, published; double close, volume; uint64_t trades; };
 struct Input { uint64_t job; std::array<float,490> observation; };
 const size_t limit;
 std::vector<Bar> bars;
 std::vector<Input> inputs;
 uint64_t tf_batches=0, tf_contexts=0, sac_calls=0, sac_rows=0;
 double tf_cuda_ms=0, sac_cuda_ms=0;

 explicit PortfolioCapture(int64_t duration):limit(size_t(duration/NS+8)*100) {
  require(duration>0&&duration<=1800*NS,"portfolio capture duration bound");
  bars.reserve(limit);
  inputs.reserve(limit);
 }
 void observe(uint32_t slot,const CanonicalBar& bar,int64_t feed,int64_t published) {
  require(bars.size()<limit,"portfolio bar capture bound");
  bars.push_back({slot,bar.ts,feed,published,bar.close,bar.volume,bar.trades});
 }
 void record(uint64_t job,const std::array<float,490>& observation) {
  require(inputs.size()<limit,"portfolio input capture bound");
  inputs.push_back({job,observation});
 }

};

}
