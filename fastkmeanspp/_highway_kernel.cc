#include "_highway_kernel.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "_highway_kernel.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace fastkmeanspp {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void HighwayDists(
    const float* x,
    const float* y,
    float* out,
    const std::size_t n,
    const std::size_t m,
    const std::size_t dim
) {
  const hn::ScalableTag<float> tag;
  const std::size_t lanes = hn::Lanes(tag);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      auto sum = hn::Zero(tag);
      std::size_t p = 0;
      for (; p + lanes <= dim; p += lanes) {
        const auto xv = hn::LoadU(tag, x + i * dim + p);
        const auto yv = hn::LoadU(tag, y + j * dim + p);
        const auto delta = hn::Sub(xv, yv);
        sum = hn::MulAdd(delta, delta, sum);
      }
      float distance = hn::ReduceSum(tag, sum);
      for (; p < dim; ++p) {
        const float delta = x[i * dim + p] - y[j * dim + p];
        distance += delta * delta;
      }
      out[i * m + j] = distance;
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace fastkmeanspp
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fastkmeanspp {
HWY_EXPORT(HighwayDists);

void highway_dists(
    const float* x,
    const float* y,
    float* out,
    const std::size_t n,
    const std::size_t m,
    const std::size_t dim
) {
  HWY_DYNAMIC_DISPATCH(HighwayDists)(x, y, out, n, m, dim);
}

}  // namespace fastkmeanspp
#endif
