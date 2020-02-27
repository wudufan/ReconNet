#include "HelicalBP.h"

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cuda_device_runtime_api.h>

using namespace std;

__global__ void bpHelicalParallelConebeam(float* pImg, cudaTextureObject_t texPrj,
		int nc, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v, float dsd, float dso,
		int nviewPerPI, float theta0, float volZ0, float zrot, int mPI, float Q)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	register float x = (ix - (nx-1) / 2.0f) * dx;
	register float y = (iy - (ny-1) / 2.0f) * dy;
	register float z = iz * dz + volZ0;

	// find the current pi segment
	int iStart = int((z / zrot * 2 * PI - PI / 4) * nviewPerPI / PI);

	register float val, imgVal;
	register float w, totalW;
	register float theta, cosTheta, sinTheta, u, v, q;
	imgVal = 0;
	for (int i = iStart; i < iStart + nviewPerPI; i++)
	{
		totalW = 0;
		val = 0;
		for (int k = i - mPI * nviewPerPI; k <= i + mPI * nviewPerPI; k+= nviewPerPI)
		{
			if (k < 0 || k >= nview)
			{
				continue;
			}

			theta = theta0 + k * PI / nviewPerPI;

			cosTheta = __cosf(theta);
			sinTheta = __sinf(theta);

			u = x * sinTheta - y * cosTheta;
//			v = 0;
			v = (z - zrot * (theta - asinf(u / dso) - theta0) / 2 / PI) * dsd / (sqrtf(dso * dso - u * u) - (x * cosTheta + y * sinTheta));

			// weighting function
			q = abs(v / dv / nv * 2.0f);
			if (q <= Q)
			{
				w = 1;
			}
			else if (q < 1)
			{
				w = __cosf((q - Q) / (1 - Q) * PI / 2);
				w *= w;
			}
			else
			{
				w = 0;
			}

			u = u / du + (nu - 1.0f) / 2.0f + off_u;
			v = v / dv + (nv - 1.0f) / 2.0f + off_v;

			val += tex3D<float>(texPrj, v + 0.5f, k + 0.5f, u + 0.5f) * w;
			totalW += w;
		}
		imgVal += val / (totalW + 1e-6f);
//		imgVal += totalW;
	}

	pImg[(ix * nz * ny + iy * nz + iz) * nc] += imgVal;

}

cHelicalBPFromParallelConebeam::cHelicalBPFromParallelConebeam(): cProjector()
{
	nview = 0;
	theta0 = 0;
	volZ0 = 0;
	zrot = 0;
	mPI = 1;
	Q = 0.5f;
}

void cHelicalBPFromParallelConebeam::Setup(int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v, float dsd, float dso,
		int nviewPerPI, float theta0, float volZ0, float zrot, int mPI, float Q)
{
	cProjector::Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz, nu, nview, nv, du, dv, off_u, off_v, dsd, dso, 3);
	this->nviewPerPI = nviewPerPI;
	this->theta0 = theta0;
	this->volZ0 = volZ0;
	this->zrot = zrot;
	this->mPI = mPI;
	this->Q = Q;
}

void cHelicalBPFromParallelConebeam::Backprojection(float* pcuImg, const float* pcuPrj, const float* pDeg)
{
	dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), nz);

	cudaMemsetAsync(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels, m_stream);
	cudaMemcpyAsync(m_pcuPrj, pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels,
			cudaMemcpyDeviceToDevice, m_stream);

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			BindPrjTex(m_pcuPrj + ib * nu * nview * nv * nChannels + ic);
			bpHelicalParallelConebeam<<<blocks, threads>>>(pcuImg + ib * nx * ny * nz * nChannels + ic,
					m_texPrj, nChannels, nx, ny, nz, dx, dy, dz, nu, nview, nv, du, dv, off_u, off_v, dsd, dso,
					nviewPerPI, theta0, volZ0, zrot, mPI, Q);
			cudaDeviceSynchronize();
		}
	}

}

extern"C" void cHelicalBPFromParallelConebeamBackprojection(float* pImg, const float* pPrj,
		int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v, float dsd, float dso,
		int nviewPerPI, float theta0, float volZ0, float zrot, int mPI = 1, float Q = 0.5)
{
	cHelicalBPFromParallelConebeam projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, du, dv, off_u, off_v, dsd, dso,
			nviewPerPI, theta0, volZ0, zrot, mPI, Q);

	float* pcuImg = NULL;
	float* pcuPrj = NULL;
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

		cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels);

		projector.SetupPrjTexture();
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);

		ostringstream oss;
		oss << "cHelicalBPFromParallelConebeamBackprojection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	projector.Backprojection(pcuImg, pcuPrj, NULL);
	cudaMemcpy(pImg, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	projector.DestroyTextures();
	cudaFree(pcuImg);
	cudaFree(pcuPrj);
}

__global__ void bpParallelFanbeam(float* pImg, cudaTextureObject_t texPrj,
		int nc, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v, float dso,
		float theta0)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	register float x = (ix - (nx-1) / 2.0f) * dx;
	register float y = (iy - (ny-1) / 2.0f) * dy;
	register float z = iz * dz;

	register float val = 0;
	register float theta, cosTheta, sinTheta, u, v;
	for (int i = 0; i < nview; i++)
	{
		theta = theta0 + i * 2 * PI / nview;

		cosTheta = __cosf(theta);
		sinTheta = __sinf(theta);

		u = x * cosTheta + y * sinTheta;
		v = z / dv;

		u = -(u / du + off_u) + (nu - 1.0f) / 2.0f;

		val += tex3D<float>(texPrj, v + 0.5f, i + 0.5f, u + 0.5f);
	}

	pImg[(ix * nz * ny + iy * nz + iz) * nc] += val;

}


void cHelicalBPFromParallelConebeam::BackprojectionParallel2D(float* pcuImg, const float* pcuPrj, const float* pDeg)
{
	dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), nz);

	cudaMemsetAsync(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels, m_stream);
	cudaMemcpyAsync(m_pcuPrj, pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels,
			cudaMemcpyDeviceToDevice, m_stream);

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			BindPrjTex(m_pcuPrj + ib * nu * nview * nv * nChannels + ic);
			bpParallelFanbeam<<<blocks, threads>>>(pcuImg + ib * nx * ny * nz * nChannels + ic,
					m_texPrj, nChannels, nx, ny, nz, dx, dy, dz, nu, nview, nv, du, dv, off_u, off_v, dso,
					theta0);
			cudaDeviceSynchronize();
		}
	}

}

extern"C" void cHelicalBPFromParallelConebeamBackprojectionParallel2D(float* pImg, const float* pPrj,
		int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v, float dso,
		float theta0)
{
	cHelicalBPFromParallelConebeam projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, du, dv, off_u, off_v, dso, dso,
			nview, theta0, 0, 0, 0, 1);

	float* pcuImg = NULL;
	float* pcuPrj = NULL;
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

		cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels);

		projector.SetupPrjTexture();
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);

		ostringstream oss;
		oss << "cHelicalBPFromParallelConebeamBackprojectionParallel2D failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	projector.BackprojectionParallel2D(pcuImg, pcuPrj, NULL);
	cudaMemcpy(pImg, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	projector.DestroyTextures();
	cudaFree(pcuImg);
	cudaFree(pcuPrj);
}
