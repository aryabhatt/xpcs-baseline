#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "mdscatter.h"

PYBIND11_MODULE (mdscatter, m) {
		m.def("cpudft", &cpu_dft, "Compute Discrete Fourier Transform at precribed q-points");
		m.def("gpudft", &gpu_dft, "Compute Discrete Fourier Transform at precribed q-points");
}
