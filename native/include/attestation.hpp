#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace shadow {
using AdmissionMac=std::array<uint8_t,32>;
AdmissionMac admission_key();
AdmissionMac admission_mac(const AdmissionMac& key,const uint8_t* bytes,size_t size);
}
