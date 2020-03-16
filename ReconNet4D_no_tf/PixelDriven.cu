#include "Projector.h"

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cuda_device_runtime_api.h>

using namespace std;

const static int zBatchBP = 5;
__global__ void bpFanKernel3D(float* pImg, cudaTextureObject_t texPrj, const float* pDeg,
		int nc, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, bool isFBP)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int izBatch = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	register float x = (ix - (nx-1) / 2.0f) * dx;
	register float y = (iy - (ny-1) / 2.0f) * dy;
	register int zStart = izBatch * zBatchBP;

	register float val[zBatchBP] = {0};
	register float cosDeg, sinDeg, rx, ry, pu, pv, a, dist;
	for (int i = 0; i < nview; i++)
	{
		cosDeg = __cosf(pDeg[i]);
		sinDeg = __sinf(pDeg[i]);
		rx =  x*cosDeg + y*sinDeg;
		ry = -x*sinDeg + y*cosDeg;
		a = atanf(rx/(ry+dso));
		if (isFBP)
		{
			dist = dso*dso / (rx*rx + (dso+ry)*(dso+ry));
		}
		else
		{
			float sin_a = fabs(__sinf(a));
			if (sin_a > 1e-6f)
			{
				dist = fminf(dy / __cosf(a), dx / sin_a);
			}
			else
			{
				dist = dy / __cosf(a);
			}
		}

		pu = -(a/da + off_a) + (nu - 1.0f) / 2.0f;

#pragma unroll
		for (int iz = 0; iz < zBatchBP; iz++)
		{
			pv = (iz + zStart - (nz-1) / 2.0f) * dz / dv + off_v + (nv - 1.0f) / 2.f;

			val[iz] += tex3D<float>(texPrj, pv + 0.5f, i + 0.5f, pu + 0.5f) * dist;
		}

	}
#pragma unroll
	for (int iz = 0; iz < zBatchBP; iz++)
	{
		if (iz + zStart < nz)
		{
			pImg[ix * nz * ny * nc + iy * nz * nc + (iz+zStart) * nc] += val[iz];
		}
	}

}

void cPixelDrivenFan::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), ceilf(nz / (float)zBatchBP));

	cudaMemsetAsync(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels, m_stream);
	cudaMemcpyAsync(m_pcuPrj, pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels,
			cudaMemcpyDeviceToDevice, m_stream);

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			BindPrjTex(m_pcuPrj + ib * nu * nview * nv * nChannels + ic);
			bpFanKernel3D<<<blocks, threads, 0, m_stream>>>(pcuImg + ib * nx * ny * nz * nChannels + ic,
					m_texPrj, pcuDeg, nChannels,
					nx, ny, nz, dx, dy, dz, nu, nview, nv, du, dv, off_u, off_v, dsd, dso, true);
			cudaDeviceSynchronize();
		}
	}
}

// single-time C interface
extern "C" void cPixelDrivenFanBackprojection(float* pImg, const float* pPrj, const float* pDeg,
		int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, int typeProjector = 0)
{
	cPixelDrivenFan projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeProjector);

	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuDeg = NULL;
	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw runtime_error("pcuPrj allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
		{
			throw runtime_error("pcuDeg allocation failed");
		}

		cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDeg, pDeg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels);

		projector.SetupPrjTexture();
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuDeg != NULL) cudaFree(pcuDeg);

		ostringstream oss;
		oss << "cPixelDrivenFanBackprojection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	projector.Backprojection(pcuImg, pcuPrj, pcuDeg);
	cudaMemcpy(pImg, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	projector.DestroyTextures();
	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuDeg);
}
