#pragma once

#include <cstddef>
#include <cstdint>

namespace fastkmeanspp {

void *create_pool(std::size_t n_jobs);

void destroy_pool(void *pool);

void dispatch_cdist(
    const float *x,
    const float *y,
    float *out,
    std::size_t n,
    std::size_t m,
    std::size_t d,
    const float *minimums,
    std::size_t minimum_stride,
    float *inertias,
    void *pool
);

void dispatch_lloyd(
    const float *x,
    float *centers,
    std::int64_t *labels,
    std::size_t n,
    std::size_t k,
    std::size_t d,
    void *pool
);

void dispatch_assign(
    const float *x,
    const float *centers,
    std::int64_t *labels,
    std::size_t n,
    std::size_t k,
    std::size_t d,
    void *pool
);

}  // namespace fastkmeanspp
