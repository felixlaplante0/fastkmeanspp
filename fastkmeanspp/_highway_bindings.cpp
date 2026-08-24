#include "_highway_kernel.h"

#include <cstddef>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style>;
using StridedFloatArray = py::array_t<float, 0>;

void check_ndim(const py::buffer_info &array, int ndim) {
  if (array.ndim != ndim) {
    throw py::value_error("expected an array with " + std::to_string(ndim) +
                          " dimensions");
  }
}

class CdistWorker {
 public:
  explicit CdistWorker(std::size_t n_jobs)
      : pool_(fastkmeanspp::create_pool(n_jobs)) {}
  ~CdistWorker() { fastkmeanspp::destroy_pool(pool_); }

  py::array_t<float> operator()(const FloatArray &x, const FloatArray &y) const {
    const auto x_info = x.request();
    const auto y_info = y.request();
    check_ndim(x_info, 2);
    check_ndim(y_info, 2);
    py::array_t<float> out({x_info.shape[0], y_info.shape[0]});
    auto out_info = out.request();
    {
      py::gil_scoped_release release;
      fastkmeanspp::dispatch_cdist(
          static_cast<const float *>(x_info.ptr),
          static_cast<const float *>(y_info.ptr),
          static_cast<float *>(out_info.ptr),
          static_cast<std::size_t>(x_info.shape[0]),
          static_cast<std::size_t>(y_info.shape[0]),
          static_cast<std::size_t>(x_info.shape[1]), nullptr, 0, nullptr,
          pool_);
    }
    return out;
  }

  py::tuple minimum(const FloatArray &x, const FloatArray &y,
                    const StridedFloatArray &minimums) const {
    const auto x_info = x.request();
    const auto y_info = y.request();
    const auto minimums_info = minimums.request();
    check_ndim(x_info, 2);
    check_ndim(y_info, 2);
    check_ndim(minimums_info, 1);
    py::array_t<float> out({x_info.shape[0], y_info.shape[0]});
    py::array_t<float> inertias(y_info.shape[0]);
    auto out_info = out.request();
    auto inertias_info = inertias.request();
    {
      py::gil_scoped_release release;
      fastkmeanspp::dispatch_cdist(
          static_cast<const float *>(x_info.ptr),
          static_cast<const float *>(y_info.ptr),
          static_cast<float *>(out_info.ptr),
          static_cast<std::size_t>(x_info.shape[0]),
          static_cast<std::size_t>(y_info.shape[0]),
          static_cast<std::size_t>(x_info.shape[1]),
          static_cast<const float *>(minimums_info.ptr),
          static_cast<std::size_t>(minimums_info.strides[0] / sizeof(float)),
          static_cast<float *>(inertias_info.ptr), pool_);
    }
    return py::make_tuple(out, inertias);
  }

 private:
  void *pool_;
};

}  // namespace

PYBIND11_MODULE(_highway, module) {
  py::class_<CdistWorker>(module, "_CdistWorker")
      .def(py::init<std::size_t>(), py::arg("n_jobs"))
      .def("__call__", &CdistWorker::operator())
      .def("minimum", &CdistWorker::minimum);
}
