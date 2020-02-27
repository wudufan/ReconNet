#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

// assuming same padding
__global__ void Variation2DKernel(float* var, const float* img, int nx, int ny, int nc, float eps)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	int ind = (ix * ny + iy) * nc;
	float x = img[ind];

	float dx = 0;
	float dy = 0;
	if (ix > 0)
	{
		dx = x - img[ind - ny * nc];
	}

	if (iy > 0)
	{
		dy = x - img[ind - nc];
	}

	var[ind] = sqrtf(dx * dx + dy * dy + eps);
}

// s1, s2 are the first and second derivatives of the surrogate function
__global__ void TVSQS2DKernel(float* s1, float* s2, const float* var, const float* img, int nx, int ny, int nc)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	int ind = (ix * ny + iy) * nc;
	float x = img[ind];
	float v = var[ind];

	float dx0 = 0;
	float dx1 = 0;
	float dy0 = 0;
	float dy1 = 0;
	float varx, vary;

	if (ix > 0)
	{
		dx0 = x - img[ind - ny * nc];
	}
	if (ix < nx - 1)
	{
		dx1 = x - img[ind + ny * nc];
		varx = var[ind + ny * nc];
	}
	else
	{
		varx = v;  	// approximate var(nx,j) with var(nx-1,j)
	}
	if (iy > 0)
	{
		dy0 = x - img[ind - nc];
	}
	if (iy < ny - 1)
	{
		dy1 = x - img[ind + nc];
		vary = var[ind + nc];
	}
	else
	{
		vary = v;	// approximate var(i, ny) with var(i, ny-1)
	}


	s1[ind] = (dx0 + dy0) / v + dx1 / varx + dy1 / vary;
	s2[ind] = 2 / v + 1 / varx + 1 / vary;

}

// get first and second order derivatives of the surrogate function
// get the variation of the image
void TVSQS2D_gpu(float* s1, float* s2, float* var, const float* img, int nb, int nx, int ny, int nc, float eps)
{
	dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), 1);
	for (int ib = 0; ib < nb; ib++)
	{
		for (int ic = 0; ic < nc; ic++)
		{
			int offset = ib * nx * ny * nc + ic;
			Variation2DKernel<<<blocks, threads>>>(var + offset, img + offset, nx, ny, nc, eps);
			TVSQS2DKernel<<<blocks, threads>>>(s1 + offset, s2 + offset, var + offset, img + offset, nx, ny, nc);
		}
	}

}

extern "C" void cTVSQS2D(float* s1, float* s2, float* var, const float* img, int nb, int nx, int ny, int nc, float eps = 1e-8f)
{
	float* pcus1 = NULL;
	float* pcus2 = NULL;
	float* pcuVar = NULL;
	float* pcuImg = NULL;

	try
	{
		int N = nb * nx * ny * nc;

		if (cudaSuccess != cudaMalloc(&pcus1, sizeof(float) * N))
		{
			throw runtime_error("pcus1 allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcus2, sizeof(float) * N))
		{
			throw runtime_error("pcus2 allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuVar, sizeof(float) * N))
		{
			throw runtime_error("pcuVar allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * N))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * N, cudaMemcpyHostToDevice);

		TVSQS2D_gpu(pcus1, pcus2, pcuVar, pcuImg, nb, nx, ny, nc, eps);

		cudaMemcpy(s1, pcus1, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(s2, pcus2, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(var, pcuVar, sizeof(float) * N, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		if (pcus1 != NULL) cudaFree(pcus1);
		if (pcus2 != NULL) cudaFree(pcus2);
		if (pcuVar != NULL) cudaFree(pcuVar);
		if (pcuImg != NULL) cudaFree(pcuImg);

		ostringstream oss;
		oss << "cTVSQS2D failed: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	cudaFree(pcus1);
	cudaFree(pcus2);
	cudaFree(pcuVar);
	cudaFree(pcuImg);

}

// assuming same padding
__global__ void Variation3DKernel(float* var, const float* img, int nx, int ny, int nz, int nc, float eps)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	int ind = (ix * ny * nz + iy * nz + iz) * nc;
	float x = img[ind];

	float dx = 0;
	float dy = 0;
	float dz = 0;
	if (ix > 0)
	{
		dx = x - img[ind - ny * nz * nc];
	}

	if (iy > 0)
	{
		dy = x - img[ind - nz * nc];
	}

	if (iz > 0)
	{
		dz = x - img[ind - nc];
	}

	var[ind] = sqrtf(dx * dx + dy * dy + dz * dz + eps);
}

// s1, s2 are the first and second derivatives of the surrogate function
__global__ void TVSQS3DKernel(float* s1, float* s2, const float* var, const float* img, int nx, int ny, int nz, int nc)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	int ind = (ix * ny * nz + iy * nz + iz) * nc;
	float x = img[ind];
	float v = var[ind];

	float dx0 = 0;
	float dx1 = 0;
	float dy0 = 0;
	float dy1 = 0;
	float dz0 = 0;
	float dz1 = 0;
	float varx, vary, varz;

	if (ix > 0)
	{
		dx0 = x - img[ind - ny * nz * nc];
	}
	if (ix < nx - 1)
	{
		dx1 = x - img[ind + ny * nz * nc];
		varx = var[ind + ny * nz * nc];
	}
	else
	{
		varx = v;  	// approximate var(nx,j) with var(nx-1,j)
	}

	if (iy > 0)
	{
		dy0 = x - img[ind - nz * nc];
	}
	if (iy < ny - 1)
	{
		dy1 = x - img[ind + nz * nc];
		vary = var[ind + nz * nc];
	}
	else
	{
		vary = v;	// approximate var(i, ny) with var(i, ny-1)
	}

	if (iz > 0)
	{
		dz0 = x - img[ind - nc];
	}
	if (iz < nz - 1)
	{
		dz1 = x - img[ind + nc];
		varz = var[ind + nc];
	}
	else
	{
		varz = v;
	}


	s1[ind] = (dx0 + dy0 + dz0) / v + dx1 / varx + dy1 / vary + dz1 / varz;
	s2[ind] = 3 / v + 1 / varx + 1 / vary + 1 / varz;

}

// get first and second order derivatives of the surrogate function
// get the variation of the image
void TVSQS3D_gpu(float* s1, float* s2, float* var, const float* img, int nb, int nx, int ny, int nz, int nc, float eps)
{
	dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), nz);
	for (int ib = 0; ib < nb; ib++)
	{
		for (int ic = 0; ic < nc; ic++)
		{
			int offset = ib * nx * ny * nz * nc + ic;
			Variation3DKernel<<<blocks, threads>>>(var + offset, img + offset, nx, ny, nz, nc, eps);
			TVSQS3DKernel<<<blocks, threads>>>(s1 + offset, s2 + offset, var + offset, img + offset, nx, ny, nz, nc);
		}
	}

}

extern "C" void cTVSQS3D(float* s1, float* s2, float* var, const float* img, int nb, int nx, int ny, int nz, int nc, float eps = 1e-8f)
{
	float* pcus1 = NULL;
	float* pcus2 = NULL;
	float* pcuVar = NULL;
	float* pcuImg = NULL;

	try
	{
		int N = nb * nx * ny * nz * nc;

		if (cudaSuccess != cudaMalloc(&pcus1, sizeof(float) * N))
		{
			throw runtime_error("pcus1 allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcus2, sizeof(float) * N))
		{
			throw runtime_error("pcus2 allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuVar, sizeof(float) * N))
		{
			throw runtime_error("pcuVar allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * N))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * N, cudaMemcpyHostToDevice);

		TVSQS3D_gpu(pcus1, pcus2, pcuVar, pcuImg, nb, nx, ny, nz, nc, eps);

		cudaMemcpy(s1, pcus1, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(s2, pcus2, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(var, pcuVar, sizeof(float) * N, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		if (pcus1 != NULL) cudaFree(pcus1);
		if (pcus2 != NULL) cudaFree(pcus2);
		if (pcuVar != NULL) cudaFree(pcuVar);
		if (pcuImg != NULL) cudaFree(pcuImg);

		ostringstream oss;
		oss << "cTVSQS3D failed: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	cudaFree(pcus1);
	cudaFree(pcus2);
	cudaFree(pcuVar);
	cudaFree(pcuImg);

}

