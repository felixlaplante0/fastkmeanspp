#include "_highway_kernel.h"

#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "_highway_kernel.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

HWY_BEFORE_NAMESPACE();
namespace fastkmeanspp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void HighwayCdist(
    const float* x,
    const float* y,
    float* out,
    const std::size_t m,
    const std::size_t d,
    const float* minimums,
    const std::size_t minimum_stride,
    float* inertias,
    const std::size_t i_begin,
    const std::size_t i_end
) {
  const hn::ScalableTag<float> tag;
  const std::size_t lanes = hn::Lanes(tag);
  if (inertias != nullptr) std::fill(inertias, inertias + m, 0.0F);
  for (std::size_t i = i_begin; i < i_end; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      auto sum = hn::Zero(tag);
      std::size_t p = 0;
      for (; p + lanes <= d; p += lanes) {
        const auto xv = hn::LoadU(tag, x + i * d + p);
        const auto yv = hn::LoadU(tag, y + j * d + p);
        const auto delta = hn::Sub(xv, yv);
        sum = hn::MulAdd(delta, delta, sum);
      }
      float distance = hn::ReduceSum(tag, sum);
      for (; p < d; ++p) {
        const float delta = x[i * d + p] - y[j * d + p];
        distance += delta * delta;
      }
      if (minimums != nullptr) {
        distance = std::min(distance, minimums[i * minimum_stride]);
      }
      out[i * m + j] = distance;
      if (inertias != nullptr) inertias[j] += distance;
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace fastkmeanspp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fastkmeanspp {
HWY_EXPORT(HighwayCdist);

namespace {

struct CdistPool {
  hwy::AlignedUniquePtr<hwy::ThreadPool> pool;
};

}  // namespace

void* highway_cdist_pool_create(
    const std::size_t n_jobs,
    const std::size_t m
) {
  const std::size_t automatic_jobs = 1 + hwy::ThreadPool::MaxThreads();
  const std::size_t requested_jobs = n_jobs == 0 ? automatic_jobs : n_jobs;
  const std::size_t effective_jobs = std::min(requested_jobs, m);
  if (effective_jobs <= 1) return nullptr;

  auto* pool = new CdistPool;
  pool->pool = hwy::MakeUniqueAligned<hwy::ThreadPool>(effective_jobs - 1);
  if (!pool->pool) {
    delete pool;
    throw std::bad_alloc();
  }
  pool->pool->SetWaitMode(hwy::PoolWaitMode::kSpin);
  return pool;
}

void highway_cdist_pool_destroy(void* pool) {
  delete static_cast<CdistPool*>(pool);
}

void highway_cdist(
    const float* x,
    const float* y,
    float* out,
    const std::size_t n,
    const std::size_t m,
    const std::size_t d,
    const float* minimums,
    const std::size_t minimum_stride,
    float* inertias,
    void* pool
) {
  const auto run = [&](const std::size_t i_begin, const std::size_t i_end,
                       float* task_inertias) {
    HWY_DYNAMIC_DISPATCH(HighwayCdist)(
        x, y, out, m, d, minimums, minimum_stride,
        task_inertias, i_begin, i_end);
  };

  if (pool == nullptr || m <= 1) {
    run(0, n, inertias);
    return;
  }

  auto& thread_pool = *static_cast<CdistPool*>(pool)->pool;
  const std::size_t num_tasks = std::min(n, 4 * thread_pool.NumWorkers());
  const std::size_t rows_per_task = (n + num_tasks - 1) / num_tasks;
  std::vector<float> partials(inertias == nullptr ? 0 : num_tasks * m);
  thread_pool.Run(0, num_tasks, [&](const std::uint64_t task, std::size_t) {
    const std::size_t i_begin = static_cast<std::size_t>(task) * rows_per_task;
    float* task_inertias = inertias == nullptr ? nullptr : partials.data() + task * m;
    run(i_begin, std::min(i_begin + rows_per_task, n), task_inertias);
  });

  if (inertias != nullptr) {
    std::fill(inertias, inertias + m, 0.0F);
    for (std::size_t task = 0; task < num_tasks; ++task) {
      for (std::size_t j = 0; j < m; ++j) {
        inertias[j] += partials[task * m + j];
      }
    }
  }
}

}  // namespace fastkmeanspp
#endif
