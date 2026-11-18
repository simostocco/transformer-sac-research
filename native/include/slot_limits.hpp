#pragma once
#include <cstddef>

namespace shadow::stage1 {
// Retained allocation bound; no scanner/admission strategy is distributed.
inline constexpr std::size_t SmallCapSlots = 100;
}
