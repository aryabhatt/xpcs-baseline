#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

namespace py = pybind11;
typedef std::complex<float> complex_t;

const complex_t COMPLX_J(0, 1);

py::array py_dft(py::array_t<float, py::array::c_style | py::array::forcecast> Pts,
				 py::array_t<float, py::array::c_style | py::array::forcecast> qVals) {

		if ((Pts.ndim() != 2) or qVals.ndim() != 2)
			throw std::runtime_error("Input arrays must be 2-D numpy arrays");

		if ((Pts.shape()[1] != 3) or (qVals.shape()[1] != 3))
			throw std::runtime_error("input arrays must of shape [N, 3]");

		/* NumPy  will allocate the buffer */
		auto result = py::array_t<complex_t>(qVals.shape()[0]);

		py::buffer_info buf1 = Pts.request();
		py::buffer_info buf2 = qVals.request();
		py::buffer_info buf3 = result.request();
		

		float * pts = (float *) buf1.ptr;
		float * qvs = (float *) buf2.ptr;
		complex_t * ft = (complex_t *) buf3.ptr;

		unsigned npts = Pts.shape()[0];
		unsigned nq = qVals.shape()[0];
#pragma omp parallel for
		for (int i = 0; i < nq; i++) {
			ft[i] = 0;
			for (int j = 0; j < npts; j++) {
				float q_r = 0.f;
				for (int k = 0; k < 3; k++) q_r += qvs[3*i + k] * pts[3*j + k];
				ft[i] += std::exp(-COMPLX_J * q_r);
			}
		}
		return result;
}
		
PYBIND11_MODULE (mdscatter, m) {
		m.def("dft", &py_dft, "Compute Discrete Fourier Transform at precribed q-points");
}
