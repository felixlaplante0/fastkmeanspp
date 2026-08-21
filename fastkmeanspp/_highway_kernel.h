#pragma once

#include <cstddef>

namespace fastkmeanspp {

void* highway_cdist_pool_create(std::size_t n_jobs, std::size_t m);

void highway_cdist_pool_destroy(void* pool);

void highway_cdist(
    const float* x,
    const float* y,
    float* out,
    std::size_t n,
    std::size_t m,
    std::size_t d,
    const float* minimums,
    std::size_t minimum_stride,
    float* inertias,
    void* pool
);

}  // namespace fastkmeanspp
