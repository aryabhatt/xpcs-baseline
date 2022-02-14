#include <cuda_runtime.h>
#include "complex.h"

__constant__ float rsq;

//const size_t N_PTS = 6144; // 2048 x 3 

void __global__ gpuDFT(unsigned npts, float * pts, unsigned nq, float * qvals, cucomplex_t * ft) {
	 
	const cucomplex_t NEG_I = make_cuFloatComplex(0.f, -1.f);
	// compute dft

	unsigned i = blockDim.x * blockIdx.x  + threadIdx.x;

	if (i < nq) {
		ft[i] = make_cuFloatComplex(0.f, 0.f);
		for (unsigned j = 0; j < npts; j++) {
			float q_dot_r = 0;
			for (unsigned k = 0; k < 3; k++) 
				q_dot_r += qvals[3 * i + k] * pts[3 * j + k];
			ft[i] = ft[i] + Cexpf(NEG_I * q_dot_r);
		}
	}
}

void cudft(unsigned npts, float * pts, unsigned nq, float * qvals,
			complex_t * output) {

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

	// device parameters
	unsigned threads = 256;
	unsigned blocks = nq / threads; 
	if (nq % threads != 0) blocks++;
	gpuDFT<<< blocks, threads >>> (npts, dpts, nq, dqvals, dft);

	// copy results back to host
	cudaMemcpy(output, dft, sizeof(complex_t) * nq, cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dpts);
	cudaFree(dqvals);
	cudaFree(dft);
}
