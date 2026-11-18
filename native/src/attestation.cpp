#include "attestation.hpp"
#include "core.hpp"
// Use OpenSSL's public stack SHA256 context API: EVP allocates contexts. This
// isolated compatibility unit suppresses OpenSSL 3 deprecation annotations.
#define OPENSSL_SUPPRESS_DEPRECATED
#include <openssl/sha.h>
#include <sys/random.h>
#include <cerrno>
namespace shadow {
AdmissionMac admission_key(){AdmissionMac key;size_t used=0;while(used<key.size()){ssize_t n=getrandom(key.data()+used,key.size()-used,0);if(n<0&&errno==EINTR)continue;require(n>0,"admission key entropy unavailable");used+=size_t(n);}return key;}
AdmissionMac admission_mac(const AdmissionMac& key,const uint8_t* bytes,size_t size){std::array<uint8_t,64> inner_pad{},outer_pad{};for(size_t i=0;i<64;++i){uint8_t k=i<key.size()?key[i]:0;inner_pad[i]=k^0x36;outer_pad[i]=k^0x5c;}AdmissionMac inner,result;SHA256_CTX ctx;
 require(SHA256_Init(&ctx)&&SHA256_Update(&ctx,inner_pad.data(),inner_pad.size())&&SHA256_Update(&ctx,bytes,size)&&SHA256_Final(inner.data(),&ctx),"admission SHA256 failure");
 require(SHA256_Init(&ctx)&&SHA256_Update(&ctx,outer_pad.data(),outer_pad.size())&&SHA256_Update(&ctx,inner.data(),inner.size())&&SHA256_Final(result.data(),&ctx),"admission HMAC failure");return result;}
}
