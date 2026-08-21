#pragma once

#include <cstddef>

namespace fastkmeanspp {

void highway_dists(
    const float* x,
    const float* y,
    float* out,
    std::size_t n,
    std::size_t m,
    std::size_t dim
);

}  // namespace fastkmeanspp
