#include "_highway_kernel.h"

#include <algorithm>
#include <cstdint>
#include <limits>
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

void cdist_kernel(
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

void lloyd_kernel(
    const float* x,
    const float* centers,
    std::int64_t* labels,
    float* sums,
    std::size_t* counts,
    const std::size_t k,
    const std::size_t d,
    const std::size_t i_begin,
    const std::size_t i_end
) {
  const hn::ScalableTag<float> tag;
  const std::size_t lanes = hn::Lanes(tag);
  for (std::size_t i = i_begin; i < i_end; ++i) {
    float best_distance = std::numeric_limits<float>::max();
    std::size_t best_cluster = 0;
    for (std::size_t j = 0; j < k; ++j) {
      auto sum = hn::Zero(tag);
      std::size_t p = 0;
      for (; p + lanes <= d; p += lanes) {
        const auto xv = hn::LoadU(tag, x + i * d + p);
        const auto center = hn::LoadU(tag, centers + j * d + p);
        const auto delta = hn::Sub(xv, center);
        sum = hn::MulAdd(delta, delta, sum);
      }
      float distance = hn::ReduceSum(tag, sum);
      for (; p < d; ++p) {
        const float delta = x[i * d + p] - centers[j * d + p];
        distance += delta * delta;
      }
      if (distance < best_distance) {
        best_distance = distance;
        best_cluster = j;
      }
    }
    labels[i] = static_cast<std::int64_t>(best_cluster);
    if (sums == nullptr) continue;

    ++counts[best_cluster];
    std::size_t p = 0;
    for (; p + lanes <= d; p += lanes) {
      const auto old_sum = hn::LoadU(tag, sums + best_cluster * d + p);
      const auto xv = hn::LoadU(tag, x + i * d + p);
      hn::StoreU(hn::Add(old_sum, xv), tag, sums + best_cluster * d + p);
    }
    for (; p < d; ++p) {
      sums[best_cluster * d + p] += x[i * d + p];
    }
  }
}

void update_kernel(
    float* centers,
    const float* sums,
    const std::size_t* counts,
    const std::size_t k,
    const std::size_t d
) {
  const hn::ScalableTag<float> tag;
  const std::size_t lanes = hn::Lanes(tag);
  for (std::size_t j = 0; j < k; ++j) {
    if (counts[j] == 0) continue;
    const auto count = hn::Set(tag, static_cast<float>(counts[j]));
    std::size_t p = 0;
    for (; p + lanes <= d; p += lanes) {
      const auto sum = hn::LoadU(tag, sums + j * d + p);
      hn::StoreU(hn::Div(sum, count), tag, centers + j * d + p);
    }
    for (; p < d; ++p) {
      centers[j * d + p] = sums[j * d + p] / counts[j];
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace fastkmeanspp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fastkmeanspp {
HWY_EXPORT(cdist_kernel);
HWY_EXPORT(lloyd_kernel);
HWY_EXPORT(update_kernel);

namespace {

struct CdistPool {
  hwy::AlignedUniquePtr<hwy::ThreadPool> pool;
};

}  // namespace

void* create_pool(std::size_t n_jobs) {
  n_jobs = n_jobs == 0 ? 1 + hwy::ThreadPool::MaxThreads() : n_jobs;
  if (n_jobs <= 1) return nullptr;

  auto* pool = new CdistPool;
  pool->pool = hwy::MakeUniqueAligned<hwy::ThreadPool>(n_jobs - 1);
  if (!pool->pool) {
    delete pool;
    throw std::bad_alloc();
  }
  pool->pool->SetWaitMode(hwy::PoolWaitMode::kSpin);
  return pool;
}

void destroy_pool(void* pool) {
  delete static_cast<CdistPool*>(pool);
}

void dispatch_cdist(
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
    HWY_DYNAMIC_DISPATCH(cdist_kernel)(
        x, y, out, m, d, minimums, minimum_stride,
        task_inertias, i_begin, i_end);
  };

  if (pool == nullptr) {
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

void dispatch_lloyd(
    const float* x,
    float* centers,
    std::int64_t* labels,
    const std::size_t n,
    const std::size_t k,
    const std::size_t d,
    void* pool
) {
  const std::size_t num_tasks = pool == nullptr
      ? 1
      : std::min(n, 4 * static_cast<CdistPool*>(pool)->pool->NumWorkers());
  const std::size_t rows_per_task = (n + num_tasks - 1) / num_tasks;
  std::vector<float> partial_sums(num_tasks * k * d, 0.0F);
  std::vector<std::size_t> partial_counts(num_tasks * k, 0);
  const auto run = [&](const std::size_t task) {
    const std::size_t i_begin = task * rows_per_task;
    HWY_DYNAMIC_DISPATCH(lloyd_kernel)(
        x, centers, labels,
        partial_sums.data() + task * k * d,
        partial_counts.data() + task * k,
        k, d, i_begin, std::min(i_begin + rows_per_task, n));
  };

  if (pool == nullptr) {
    run(0);
  } else {
    auto& thread_pool = *static_cast<CdistPool*>(pool)->pool;
    thread_pool.Run(0, num_tasks, [&](const std::uint64_t task, std::size_t) {
      run(static_cast<std::size_t>(task));
    });
  }

  std::vector<float> sums(k * d, 0.0F);
  std::vector<std::size_t> counts(k, 0);
  for (std::size_t task = 0; task < num_tasks; ++task) {
    for (std::size_t j = 0; j < k; ++j) {
      counts[j] += partial_counts[task * k + j];
      for (std::size_t p = 0; p < d; ++p) {
        sums[j * d + p] += partial_sums[task * k * d + j * d + p];
      }
    }
  }
  HWY_DYNAMIC_DISPATCH(update_kernel)(centers, sums.data(), counts.data(), k, d);
}

void dispatch_assign(
    const float* x,
    const float* centers,
    std::int64_t* labels,
    const std::size_t n,
    const std::size_t k,
    const std::size_t d,
    void* pool
) {
  const auto run = [&](const std::size_t i_begin, const std::size_t i_end) {
    HWY_DYNAMIC_DISPATCH(lloyd_kernel)(
        x, centers, labels, nullptr, nullptr, k, d, i_begin, i_end);
  };
  if (pool == nullptr) {
    run(0, n);
    return;
  }

  auto& thread_pool = *static_cast<CdistPool*>(pool)->pool;
  const std::size_t num_tasks = std::min(n, 4 * thread_pool.NumWorkers());
  const std::size_t rows_per_task = (n + num_tasks - 1) / num_tasks;
  thread_pool.Run(0, num_tasks, [&](const std::uint64_t task, std::size_t) {
    const std::size_t i_begin = static_cast<std::size_t>(task) * rows_per_task;
    run(i_begin, std::min(i_begin + rows_per_task, n));
  });
}

}  // namespace fastkmeanspp
#endif
