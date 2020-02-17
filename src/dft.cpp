#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "complex.h"
#include "mdscatter.h"

namespace py = pybind11;

py::array cpu_dft(np_array_t Pts, np_array_t qVals) {

    if ((Pts.ndim() != 2) or qVals.ndim() != 2)
        throw std::runtime_error("Input arrays must be 2-D numpy arrays");

    if ((Pts.shape()[1] != 3) or (qVals.shape()[1] != 3))
        throw std::runtime_error("input arrays must of shape [N, 3]");

    /* size of input arrays */
    unsigned npts = Pts.shape()[0];
    unsigned nq = qVals.shape()[0];

    /* NumPy  will allocate the buffer */
    auto result = py::array_t<complex_t>(nq);

    py::buffer_info buf1 = Pts.request();
    py::buffer_info buf2 = qVals.request();
    py::buffer_info buf3 = result.request();
		
    float * pts = (float *) buf1.ptr;
    float * qvs = (float *) buf2.ptr;
    complex_t * ft = (complex_t *) buf3.ptr;

    #pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        ft[i] = COMPLX_ZERO;
        for (int j = 0; j < npts; j++) {
            float d = 0;
            float q_dot_r = 0.f;
            for (int k = 0; k < 3; k++)
                q_dot_r += qvs[3*i + k] * pts[3*j + k];
            ft[i] += std::exp(-COMPLX_J * q_dot_r);
        }
    }
    return result;
}

