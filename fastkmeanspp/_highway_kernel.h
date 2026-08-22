#pragma once

#include <cstddef>

namespace fastkmeanspp {

void* create_pool(std::size_t n_jobs);

void destroy_pool(void* pool);

void run_cdist(
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
