#ifndef MD_SCATTER__H
#define MD_SCATTER__H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
typedef py::array_t<float, py::array::c_style | py::array::forcecast> np_array_t;

py::array cpu_dft(np_array_t, np_array_t);
py::array gpu_dft(np_array_t, np_array_t);

#endif // MD_SCATTER__H
