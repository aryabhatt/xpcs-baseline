#include <cuda_runtime.h>
#include "complex.h"

void __global__ gpuDFT(unsigned npts, float * pts, unsigned nq, float * qvals, cucomplex_t * ft) {
	/* lucky we have excatly 4000 particles. 4000 X 3 should fit on most modern GPUs.
     * If more particles are needed in future, we need to do some memory maneuvering gymnastics.
	 */
	const cucomplex_t NEG_I = make_cuFloatComplex(0.f, -1.f);
	// compute dft
	unsigned i = blockDim.x * blockIdx.x  + threadIdx.x;
	if (i < nq) {
		ft[i] = make_cuFloatComplex(0.f, 0.f);
		for (unsigned j = 0; j < npts; j++) {
			float q_r = 0;
			for (unsigned k = 0; k < 3; k++) 
				q_r += qvals[3 * i + k] * pts[3 * j + k];
			ft[i] = ft[i] + Cexpf(NEG_I * q_r);
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
