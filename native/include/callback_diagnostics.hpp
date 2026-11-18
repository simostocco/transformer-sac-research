#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
namespace shadow::stage1 {
struct SdkCallbackMetadata {
 bool available=false,date_available=false,clock_available=false,trade_clock_available=false;
 uint32_t message_type=0,date=0,trade_date=0;int status=0,status_data=0;uint64_t raw_clock=0,raw_trade_clock=0;
 const char* string_field=nullptr;const char* string_error=nullptr;int64_t string_length=-1;uint32_t string_bound=0;bool string_length_is_lower_bound=false;
 const char* operation=nullptr;uint32_t line=0;
};
struct CallbackFailureDiagnostic {
 bool available=false,message_available=false,truncated=false,sdk_decoder=false;
 std::array<char,512> message{};std::array<char,96> category{},type{};
 std::array<char,128> operation{};
 SdkCallbackMetadata sdk{};
 uint64_t sequence=0,session=0,epoch=0,generation=0;int lifecycle=0;
 int64_t monotonic_ns=0,wall_ns=0,source_feed_ns=0,last_source_feed_ns=0;bool source_coordinate_available=false;
};
// Failure path only. Copy/redact directly into fixed storage; no formatting,
// allocation, I/O or new synchronization in the callback.
template<size_t N>bool diagnostic_copy(std::array<char,N>& out,const char* in,std::span<const std::string> secrets={})noexcept{
 size_t at=0;if(!in)return false;
 while(*in&&at<N-1){const std::string* match=nullptr;
  for(auto& secret:secrets)if(!secret.empty()&&std::strncmp(in,secret.c_str(),secret.size())==0&&(!match||secret.size()>match->size()))match=&secret;
  if(match){constexpr char replacement[]="[REDACTED]";for(char c:replacement){if(!c)break;if(at==N-1){out[at]=0;return true;}out[at++]=c;}in+=match->size();}
  else out[at++]=*in++;
 }out[at]=0;return *in!=0;
}
struct InitializationDiagnostic {SdkCallbackMetadata sdk{};uint64_t sequence=0;int64_t monotonic_ns=0,wall_ns=0;};
struct InitializationJournal {std::array<InitializationDiagnostic,16> entries{};size_t count=0;bool truncated=false,market_seen=false;};
}
