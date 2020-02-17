#include <cuda_runtime.h>
#include "complex.h"
#include "mdscatter.h"

__constant__ float rsq;


//const size_t N_PTS = 6144; // 2048 x 3 
const int N_PTS = 512;

void __global__ gpuDFT(unsigned npts, float * pts, unsigned nq, float * qvals, cucomplex_t * ft) {

    const cucomplex_t NEG_I = make_cuFloatComplex(0.f, -1.f);

	int idx = blockDim.x * blockIdx.x  + threadIdx.x;

    __shared__ float shamem [N_PTS * 3];
	int nfills =  npts / N_PTS;
    if (npts % N_PTS) nfills++;

    int np = N_PTS;
    for (int ifill = 0; ifill < nfills; ifill++) {
        int pnt_idx = ifill * N_PTS + threadIdx.x;
        if ((ifill + 1) * N_PTS > npts) np = npts % N_PTS; 

        if (pnt_idx < npts ) {
            for (int k = 0; k < 3; k++) 
                shamem[threadIdx.x * 3 + k] = pts[pnt_idx * 3 + k];
        }

            // compute dft
        if (idx < nq) {
            for (unsigned j = 0; j < np; j++) {
			    float q_r = 0;
			    for (unsigned k = 0; k < 3; k++) 
				    q_r += qvals[3 * idx + k] * shamem[3 * j + k];
			    ft[idx] = ft[idx] + Cexpf(NEG_I * q_r); 
		    }
	    }
    }
}

py::array gpu_dft(np_array_t Pts, np_array_t qVals) {

    if ((Pts.ndim() != 2) or qVals.ndim() != 2)
        throw std::runtime_error("Input arrays must be 2-D numpy arrays");

    if ((Pts.shape()[1] != 3) or (qVals.shape()[1] != 3))
        throw std::runtime_error("input arrays must of shape [N, 3]");

    /* size of input arrays */
    unsigned npts = Pts.shape()[0];
    unsigned nq = qVals.shape()[0];

    /* NumPy  will allocate the buffer */
    auto result = py::array_t<complex_t>(nq);

    float * pts = (float *) Pts.request().ptr;
    float * qvals = (float *) qVals.request().ptr;
    complex_t * ft = (complex_t *) result.request().ptr;
	
    // copy beam-radius (squared) to constant memory
    //float brsq = beam_radius * beam_radius;
    //cudaMemcpyToSymbol(rsq, &brsq, sizeof(float), 0, cudaMemcpyHostToDevice);

	// allocate memory on device
	float * dpts, * dqvals;
	cudaMalloc((void **) &dpts, sizeof(float) * npts * 3);
	cudaMalloc((void **) &dqvals, sizeof(float) * nq * 3);

    // copy arrays to device memory
	cudaMemcpy(dpts, pts, sizeof(float) * 3 * npts, cudaMemcpyHostToDevice);
	cudaMemcpy(dqvals, qvals, sizeof(float) * 3 * nq, cudaMemcpyHostToDevice);

	// allocate memory for output
	cucomplex_t * dft = NULL;
	cudaMalloc((void **) &dft, sizeof(cucomplex_t) * nq);
    cudaMemset(dft, sizeof(cucomplex_t) * nq, 0);

	// device parameters
	unsigned threads = N_PTS;
	unsigned blocks = nq / threads; 
	if (nq % threads != 0) blocks++;
	gpuDFT<<< blocks, threads >>> (npts, dpts, nq, dqvals, dft);

	// copy results back to host
	cudaMemcpy(ft, dft, sizeof(complex_t) * nq, cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dpts);
	cudaFree(dqvals);
	cudaFree(dft);
    return result;
}
