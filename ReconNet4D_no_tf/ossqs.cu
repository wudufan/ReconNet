#include "Projector.h"
#include "cudaMath.h"

#include <cuda_device_runtime_api.h>

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

extern "C" void cOSSQS(float* res, const float* img, const float* prj,
		const float* normImg, const float* wprj,
		const float* degs, float lb, float ub, int nviewPerSubset, int nSubsets,
		int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, int typeProjector = 0, float lambda = 1)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuFP = NULL;
	float* pcuBP = NULL;
	float* pcuNormImg = NULL;
	float* pcuWPrj = NULL;
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
		if (cudaSuccess != cudaMalloc(&pcuFP, sizeof(float) * nBatches * nu * nviewPerSubset * nv * nChannels))
		{
			throw runtime_error("pcuFP allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuBP, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw runtime_error("pcuBP allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuNormImg, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw runtime_error("pcuNormImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuWPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw runtime_error("pcuWPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nviewPerSubset * nSubsets))
		{
			throw runtime_error("pcuDegs allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuPrj, prj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuWPrj, wprj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuNormImg, normImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDeg, degs, sizeof(float) * nview, cudaMemcpyHostToDevice);
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuFP != NULL) cudaFree(pcuFP);
		if (pcuBP != NULL) cudaFree(pcuBP);
		if (pcuNormImg != NULL) cudaFree(pcuNormImg);
		if (pcuWPrj != NULL) cudaFree(pcuWPrj);
		if (pcuDeg != NULL) cudaFree(pcuDeg);

		ostringstream oss;
		oss << "cOSSQS() failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	SiddonFan projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nviewPerSubset, nv, da, dv, off_a, off_v, dsd, dso, typeProjector);

	dim3 threads(32, 16, 1);
	dim3 blocksImg(ceilf(ny * nz * nChannels / (float)threads.x),
			ceilf(nx * nBatches / (float)threads.y), 1);
	dim3 blocksPrj(ceilf(nviewPerSubset * nv * nChannels / (float)threads.x),
			ceilf(nu * nBatches / (float)threads.y), 1);

	int startView = 0;
	for (int iSubset = 0; iSubset < nSubsets; iSubset++)
	{
		int nviewCurrent = nviewPerSubset;
		if (startView + nviewPerSubset > nview)
		{
			nviewCurrent = nview - nviewPerSubset;
		}
		projector.nview = nviewCurrent;

		// Ax
		cudaMemset(pcuFP, 0, sizeof(float) * nBatches * nu * nviewPerSubset * nv * nChannels);
		projector.Projection(pcuImg, pcuFP, pcuDeg + startView);

		// Ax - b
		Minus2D<<<blocksPrj, threads>>>(pcuFP, pcuFP, pcuPrj + startView * nv * nChannels,
				nviewCurrent * nv * nChannels, nu * nBatches,
				nviewPerSubset * nv * nChannels,
				nviewPerSubset * nv * nChannels,
				nview * nv * nChannels);

//		float* fp = new float [nu * nview * nv];
//		cudaMemcpy(fp, pcuPrj, sizeof(float) * nu * nview * nv, cudaMemcpyDeviceToHost);
//
//		delete [] fp;

		// w(Ax -b)
		Multiply2D<<<blocksPrj, threads>>>(pcuFP, pcuFP, pcuWPrj + startView * nv * nChannels,
				nviewCurrent * nv * nChannels, nu * nBatches,
				nviewPerSubset * nv * nChannels,
				nviewPerSubset * nv * nChannels,
				nview * nv * nChannels);

		// in case nview is not times of nviewPerSubset
		if (nviewCurrent < nviewPerSubset)
		{
			cudaMemset(pcuFP + nviewCurrent, 0,
					sizeof(float) * (nviewPerSubset - nviewCurrent) * nBatches * nu * nv * nChannels);
		}

		// A'w(Ax-b)
		cudaMemset(pcuBP, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels);
		projector.Backprojection(pcuBP, pcuFP, pcuDeg + startView);
		// A'w(Ax-b) / (A'wA1)
		Divide2D<<<blocksImg, threads>>>(pcuBP, pcuBP, pcuNormImg, ny * nz * nChannels, nx * nBatches,
				ny * nz * nChannels, ny * nz * nChannels, ny * nz * nChannels, 1e-6f);
		// x - l * M * A'w(Ax-b) / (A'wA1)
		Minus2D<<<blocksImg, threads>>>(pcuImg, pcuImg, pcuBP, ny * nz * nChannels, nx * nBatches,
				ny * nz * nChannels, ny * nz * nChannels, ny * nz * nChannels, lambda * nSubsets);

		// crop the value for better stability
		if (lb < ub)
		{
			ValueCrop2D<<<blocksImg, threads>>>(pcuImg, pcuImg, lb, ub, ny * nz * nChannels, nx * nBatches,
					ny * nz * nChannels, ny * nz * nChannels);
		}

		startView += nviewCurrent;
	}

	cudaMemcpy(res, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuFP);
	cudaFree(pcuBP);
	cudaFree(pcuNormImg);
	cudaFree(pcuWPrj);
	cudaFree(pcuDeg);

}

extern "C" void cOSSQSConeAbitrary(float* res, const float* img, const float* prj,
		const float* normImg, const float* mask,
		const float* detCenter, const float* detU, const float* detV, const float* src,
		float lb, float ub,
		int nviewPerSubset, int nSubsets, int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v,
		int typeProjector = 2, float lambda = 1)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuFP = NULL;
	float* pcuBP = NULL;
	float* pcuNormImg = NULL;
	float* pcuMask = NULL;
	float* pcuDetCenter = NULL;
	float* pcuDetU = NULL;
	float* pcuDetV = NULL;
	float* pcuSrc = NULL;

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
		if (cudaSuccess != cudaMalloc(&pcuFP, sizeof(float) * nBatches * nu * nviewPerSubset * nv * nChannels))
		{
			throw runtime_error("pcuFP allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuBP, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw runtime_error("pcuBP allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuNormImg, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw runtime_error("pcuNormImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuMask, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw runtime_error("pcuMask allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetCenter, sizeof(float3) * nview))
		{
			throw runtime_error("pcuDetCenter allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetU, sizeof(float3) * nview))
		{
			throw runtime_error("pcuDetU allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetV, sizeof(float3) * nview))
		{
			throw runtime_error("pcuDetV allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuSrc, sizeof(float3) * nview))
		{
			throw runtime_error("pcuSrc allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuPrj, prj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuNormImg, normImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuMask, mask, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetU, detU, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetV, detV, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuSrc, src, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuFP != NULL) cudaFree(pcuFP);
		if (pcuBP != NULL) cudaFree(pcuBP);
		if (pcuNormImg != NULL) cudaFree(pcuNormImg);
		if (pcuMask != NULL) cudaFree(pcuMask);
		if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
		if (pcuDetU != NULL) cudaFree(pcuDetU);
		if (pcuDetV != NULL) cudaFree(pcuDetV);
		if (pcuSrc != NULL) cudaFree(pcuSrc);

		ostringstream oss;
		oss << "cOSSQS() failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	SiddonCone projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nviewPerSubset, nv, du, dv, off_u, off_v, 0, 0, typeProjector);

	dim3 threads(32, 16, 1);
	dim3 blocksImg(ceilf(ny * nz * nChannels / (float)threads.x),
			ceilf(nx * nBatches / (float)threads.y), 1);
	dim3 blocksPrj(ceilf(nviewPerSubset * nv * nChannels / (float)threads.x),
			ceilf(nu * nBatches / (float)threads.y), 1);

	int startView = 0;
	for (int iSubset = 0; iSubset < nSubsets; iSubset++)
	{
		int nviewCurrent = nviewPerSubset;
		if (startView + nviewPerSubset > nview)
		{
			nviewCurrent = nview - nviewPerSubset;
		}
		projector.nview = nviewCurrent;

		// Ax
		cudaMemset(pcuFP, 0, sizeof(float) * nBatches * nu * nviewPerSubset * nv * nChannels);
		projector.ProjectionAbitrary(pcuImg, pcuFP, pcuDetCenter + 3 * startView,
				pcuDetU + 3 * startView, pcuDetV + 3 * startView, pcuSrc + 3 * startView);

		// Ax - b
		Minus2D<<<blocksPrj, threads>>>(pcuFP, pcuFP, pcuPrj + startView * nv * nChannels,
				nviewCurrent * nv * nChannels, nu * nBatches,
				nviewPerSubset * nv * nChannels,
				nviewPerSubset * nv * nChannels,
				nview * nv * nChannels);

		// in case nview is not times of nviewPerSubset
		if (nviewCurrent < nviewPerSubset)
		{
			cudaMemset(pcuFP + nviewCurrent, 0,
					sizeof(float) * (nviewPerSubset - nviewCurrent) * nBatches * nu * nv * nChannels);
		}

		// A'(Ax-b)
		cudaMemset(pcuBP, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels);
		projector.BackprojectionAbitrary(pcuBP, pcuFP, pcuDetCenter + 3 * startView,
				pcuDetU + 3 * startView, pcuDetV + 3 * startView, pcuSrc + 3 * startView);
		// A'(Ax-b) / (A'A1)
		Divide2D<<<blocksImg, threads>>>(pcuBP, pcuBP, pcuNormImg, ny * nz * nChannels, nx * nBatches,
				ny * nz * nChannels, ny * nz * nChannels, ny * nz * nChannels, 1e-6f);

		// x - l * M * A'(Ax-b) / (A'A1)
		Minus2D<<<blocksImg, threads>>>(pcuImg, pcuImg, pcuBP, ny * nz * nChannels, nx * nBatches,
				ny * nz * nChannels, ny * nz * nChannels, ny * nz * nChannels, lambda * nSubsets);

		// x = x * m
		Multiply2D<<<blocksImg, threads>>>(pcuImg, pcuImg, pcuMask, ny * nz * nChannels, nx * nBatches,
				ny * nz * nChannels, ny * nz * nChannels, ny * nz * nChannels);

		// crop the value for better stability
		if (lb < ub)
		{
			ValueCrop2D<<<blocksImg, threads>>>(pcuImg, pcuImg, lb, ub, ny * nz * nChannels, nx * nBatches,
					ny * nz * nChannels, ny * nz * nChannels);
		}

		startView += nviewCurrent;
	}

	cudaMemcpy(res, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuFP);
	cudaFree(pcuBP);
	cudaFree(pcuNormImg);
	cudaFree(pcuMask);
	cudaFree(pcuDetCenter);
	cudaFree(pcuDetU);
	cudaFree(pcuDetV);
	cudaFree(pcuSrc);

}
