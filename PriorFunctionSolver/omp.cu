#include <cusolverDn.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <iostream>
#include <sstream>

using namespace std;

// see Tropp, J.A. and Gilbert, A.C., 2007.
// Signal recovery from random measurements via orthogonal matching pursuit. IEEE Transactions on information theory, 53(12), pp.4655-4666.

// step 2: find i = argmax<r, dict_i>
// inds - to store the indices of selected atoms, size L * cnt
// converged - if the patch's omp is converged (> 0), size cnt
// r - residue of the patches, size n * cnt
// D - the dictionary, size n * d, column major
// t - current iteration
// L - sparse level
// cnt - number of vectors
// n - length of each vector
// d - number of atoms in D
__global__ void FindIndicesOfMaxResidueDot(int* inds, const int* converged, const float* r, const float* D,
		int t, int L, int n, int d, int cnt)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= cnt)
	{
		return;
	}
	if (converged[ix])
	{
		return;
	}

	r += ix * n;

	float maxRes = -1;
	int maxInd = 0;

	for (int i = 0; i < d; i++)
	{
		float sum = 0;
		for (int j = 0; j < n; j++)
		{
			sum += r[j] * D[i * n + j];
		}

		if (fabsf(sum) > maxRes)
		{
			maxRes = fabsf(sum);
			maxInd = i;
		}
	}

	inds[ix * L + t] = maxInd;

}

// Update the Hermitian matrices DtD
// DtDs_t - the DtDs to be updated, already stored the DtDs_(t-1), size L * L * cnt, only upper half is stored, column major
// inds - the indices of selected atoms, size L * cnt
// converged - if a patch is converged, skip the calculation, size cnt
// DtD - the full DtD matrix, of size d * d, column major
// t - current iteration
// L - sparse level
// d - number of atoms
// cnt - number of patches
__global__ void UpdateDtDs(float* DtDs_t, const int* inds, const int* converged, const float* DtD, int t, int L, int d, int cnt)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= cnt)
	{
		return;
	}

	inds += ix * L;
	DtDs_t += ix * L * L + t * L; // go to new column

	if (converged[ix])
	{
		DtDs_t[t] = 1;	// make sure the matrix is positive definite, to avoid any problem in cholesky factorization.
		return;
	}

	int newInd = inds[t];
	DtD += newInd * d;  // go to column DtD[:, newInd]

	// add the new column: DtDs_t[i, t] = DtD[inds[i], inds[t]]
	for (int i = 0; i < t + 1; i++)
	{
		DtDs_t[i] = DtD[inds[i]];
	}
}

// Update the residue matrix Dtv
// Dtvs_t - the residue Dtvs to be updated, size L * cnt
// inds - the indices of selected atoms, size L * cnt
// converged - if a patch is converged, skip the calculation, size cnt
// D - the dictionary, size n * d * cnt, column major
// v - the data vectors, size n * cnt
// t - current iteration
// L - sparse level
// n - length of vector
// d - number of atoms
// cnt - number of patches
__global__ void UpdateDtvs(float* Dtvs_t, const int* inds, const int* converged, const float* D, const float* v,
		int t, int L, int n, int d, int cnt)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= cnt)
	{
		return;
	}

	if (converged[ix])
	{
		return;
	}

	inds += ix * L;
	v += ix * n;
	D += inds[t] * n;  // column inds[t] in dictionary D

	float sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += D[i] * v[i];
	}

	Dtvs_t[ix * L + t] = sum;

}

// update a_t = D_t*x_t, and r_t = v - a_t
// a - the approximation to v, size n * cnt
// r - the residues, size n * cnt
// x - the solution to argmin|D_t*x - v|, size L * cnt
// inds - the indices of selected atoms, size L * cnt
// converged - if a patch is converged, skip the calculation, size cnt
// D - the dictionary, size n * d * cnt, column major
// v - the data vectors, size n * cnt
// t - current iteration
// L - sparse level
// n - length of vector
// d - number of atoms
// cnt - number of patches
__global__ void UpdateApproximationAndResidue(float* a, float* r, const float* x, const int* inds, const int* converged, const float* D, const float* v,
		int t, int L, int n, int d, int cnt)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= cnt)
	{
		return;
	}

	if (converged[ix])
	{
		return;
	}

	a += ix * n;
	r += ix * n;
	x += ix * L;
	inds += ix * L;
	v += ix * n;

	for (int j = 0; j < n; j++)
	{
		a[j] = 0;
	}

	// calculate approximations
	for (int i = 0; i < t + 1; i++)
	{
		const float* Di = D + inds[i] * n;
		float xt = x[i];
		for (int j = 0; j < n; j++)
		{
			a[j] += Di[j] * xt;
		}
	}

	// calculate the residue
	for (int j = 0; j < n; j++)
	{
		r[j] = v[j] - a[j];
	}
}

// Mark as converged if |r|^2 < thresh
// converged - if the patch is converged, size cnt
// r - the residues, size n * cnt
// n - length of residue
// cnt - number of patches
// thresh - threshold for convergence
__global__ void UpdateConverge(int* converged, const float* r, int n, int cnt, float thresh)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= cnt)
	{
		return;
	}
	if (converged[ix])
	{
		return;
	}

	r += ix * n;

	float sum = 0;
	for (int j = 0; j < n; j++)
	{
		sum += r[j] * r[j];
	}

	if (sum < thresh)
	{
		converged[ix] = 1;
	}

}

__global__ void InitPtr(float** ptr, float* array, int stride, int cnt)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix >= cnt)
	{
		return;
	}

	ptr[ix] = array + stride * ix;
}

// orthogonal matching pursuit on gpu, execute multiple patches simultaneously.
// All the input and output buffers are on GPU
// a - the approximation patches, size n * cnt
// r - the residue patches, size n * cnt
// v - the original patches, size n * cnt
// D - the dictionary, size n * d, column major
// DtD - Dt*D, size d * d, column major
// L - sparse level
// n - number of elements per patch
// d - number of atoms in dictionary
// cnt - number of patches
// thresh - threshold to determine if a patch is approximated well enough.
void omp_gpu(float* a, float* r, const float *v, const float* D, const float* DtD, int L, int n, int d, int cnt, float thresh = 1e-16)
{
	int* inds = NULL;
	int* converged = NULL;
	float* DtDs_t = NULL;
	float* Dtvs_t = NULL;
	float* A = NULL;
	float* B = NULL;
	float** ptrA = NULL;
	float** ptrB = NULL;
	cusolverDnHandle_t handle = NULL;
	int* devinfos = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&inds, sizeof(int) * L * cnt))
		{
			throw runtime_error("inds allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&converged, sizeof(int) * cnt))
		{
			throw runtime_error("converged allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&DtDs_t, sizeof(float) * L * L * cnt))
		{
			throw runtime_error("DtDs_t allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&Dtvs_t, sizeof(float) * L * cnt))
		{
			throw runtime_error("Dtvs_t allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&devinfos, sizeof(int) * cnt))
		{
			throw runtime_error("devinfos allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&A, sizeof(float) * L * L * cnt))
		{
			throw runtime_error("A allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&B, sizeof(float) * L * cnt))
		{
			throw runtime_error("B allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&ptrA, sizeof(float*) * cnt))
		{
			throw runtime_error("ptrA allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&ptrB, sizeof(float*) * cnt))
		{
			throw runtime_error("ptrB allocation failed");
		}


		// initialize memory
		cudaMemset(inds, 0, sizeof(int) * L * cnt);
		cudaMemset(converged, 0, sizeof(int) * cnt);
		cudaMemset(DtDs_t, 0, sizeof(float) * L * L * cnt);
		cudaMemset(Dtvs_t, 0, sizeof(float) * L * cnt);

		cudaMemcpy(r, v, sizeof(float) * n * cnt, cudaMemcpyDeviceToDevice);

		dim3 threads(256, 1, 1);
		dim3 blocks(ceilf(cnt / (float)threads.x), 1, 1);
		InitPtr<<<blocks, threads>>>(ptrA, A, L * L, cnt);
		InitPtr<<<blocks, threads>>>(ptrB, B, L, cnt);

		// initialize cuSolver
		if (CUSOLVER_STATUS_SUCCESS != cusolverDnCreate(&handle))
		{
			throw runtime_error("cusolverDnHandle creation failed");
		}


		for (int t = 0; t < L; t++)
		{
			// find i = argmax<r, D_i>
			FindIndicesOfMaxResidueDot<<<blocks, threads>>>(inds, converged, r, D, t, L, n, d, cnt);
			cudaDeviceSynchronize();

			// x = argmin|Dt*x - v|
			UpdateDtDs<<<blocks, threads>>>(DtDs_t, inds, converged, DtD, t, L, d, cnt);
			UpdateDtvs<<<blocks, threads>>>(Dtvs_t, inds, converged, D, v, t, L, n, d, cnt);
			cudaDeviceSynchronize();

			cudaMemcpy(A, DtDs_t, sizeof(float) * L * L * cnt, cudaMemcpyDeviceToDevice);
			cudaMemcpy(B, Dtvs_t, sizeof(float) * L * cnt, cudaMemcpyDeviceToDevice);
			if (CUSOLVER_STATUS_SUCCESS != cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_UPPER, t+1, ptrA, L, devinfos, cnt))
			{
				throw runtime_error("cusolverDnSpotrfBatched failed");
			}
			cudaDeviceSynchronize();

			if (CUSOLVER_STATUS_SUCCESS != cusolverDnSpotrsBatched(handle, CUBLAS_FILL_MODE_UPPER, t+1, 1, ptrA, L, ptrB, L, devinfos, cnt))
			{
				throw runtime_error("cusolverDnSpotrsBatched failed");
			}
			cudaDeviceSynchronize();

			// a = Dt * x, r = v - a
			UpdateApproximationAndResidue<<<blocks, threads>>>(a, r, B, inds, converged, D, v, t, L, n, d, cnt);
			cudaDeviceSynchronize();

			UpdateConverge<<<blocks, threads>>>(converged, r, n, cnt, thresh);
			cudaDeviceSynchronize();

		}
	}
	catch (exception &e)
	{
		if (inds != NULL) cudaFree(inds);
		if (converged != NULL) cudaFree(converged);
		if (DtDs_t != NULL) cudaFree(DtDs_t);
		if (Dtvs_t != NULL) cudaFree(Dtvs_t);
		if (A != NULL) cudaFree(A);
		if (B != NULL) cudaFree(B);
		if (ptrA != NULL) cudaFree(ptrA);
		if (ptrB != NULL) cudaFree(ptrB);
		if (devinfos != NULL) cudaFree(devinfos);
		if (handle != NULL) cusolverDnDestroy(handle);

		ostringstream oss;
		oss << "omp_gpu failed: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	cudaFree(inds);
	cudaFree(converged);
	cudaFree(DtDs_t);
	cudaFree(Dtvs_t);
	cudaFree(A);
	cudaFree(B);
	cudaFree(ptrA);
	cudaFree(ptrB);
	cudaFree(devinfos);
	cusolverDnDestroy(handle);

}

// orthogonal matching pursuit on gpu, execute multiple patches simultaneously.
// All the input and output buffers are on CPU
// a - the approximation patches, size n * cnt
// r - the residue patches, size n * cnt
// v - the original patches, size n * cnt
// D - the dictionary, size n * d, column major
// DtD - Dt*D, size d * d, column major
// L - sparse level
// n - number of elements per patch
// d - number of atoms in dictionary
// cnt - number of patches
// thresh - threshold to determine if a patch is approximated well enough.
extern "C" void cOrthogonalMatchingPursuit(float* a, float* r, const float* v, const float* D, const float* DtD,
		int L, int n, int d, int cnt, float thresh = 1e-16)
{
	float* pcua = NULL;
	float* pcur = NULL;
	float* pcuv = NULL;
	float* pcuD = NULL;
	float* pcuDtD = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcua, sizeof(float) * n * cnt))
		{
			throw runtime_error("pcua allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcur, sizeof(float) * n * cnt))
		{
			throw runtime_error("pcur allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuv, sizeof(float) * n * cnt))
		{
			throw runtime_error("pcuv allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuD, sizeof(float) * n * d))
		{
			throw runtime_error("pcuD allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDtD, sizeof(float) * d * d))
		{
			throw runtime_error("pcuDtD allocation failed");
		}

		cudaMemcpy(pcuv, v, sizeof(float) * n * cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuD, D, sizeof(float) * n * d, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDtD, DtD, sizeof(float) * d * d, cudaMemcpyHostToDevice);

		omp_gpu(pcua, pcur, pcuv, pcuD, pcuDtD, L, n, d, cnt, thresh);

		cudaMemcpy(a, pcua, sizeof(float) * n * cnt, cudaMemcpyDeviceToHost);
		cudaMemcpy(r, pcur, sizeof(float) * n * cnt, cudaMemcpyDeviceToHost);
	}
	catch (exception &e)
	{
		if (pcua != NULL) cudaFree(pcua);
		if (pcur != NULL) cudaFree(pcur);
		if (pcuv != NULL) cudaFree(pcuv);
		if (pcuD != NULL) cudaFree(pcuD);
		if (pcuDtD != NULL) cudaFree(pcuDtD);

		ostringstream oss;
		oss << "OrthogonalMatchingPursuit failed: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());

	}

	cudaFree(pcua);
	cudaFree(pcur);
	cudaFree(pcuv);
	cudaFree(pcuD);
	cudaFree(pcuDtD);

}




