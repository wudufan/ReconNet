#include "Siddon.h"
#include "Projector.h"
#include "cudaMath.h"

#include <stdexcept>
#include <iostream>
#include <sstream>

using namespace std;

__device__ __host__ void GetSrcDstForParallel(float3& src, float3& dst, float u, float v,
		const float3& detCenter, const float3& detU, const float3& detV, const float3& invRayDir,
		const Grid grid)
{
	dst = make_float3(detCenter.x + detU.x * u + detV.x * v,
		detCenter.y + detU.y * u + detV.y * v,
		detCenter.z + detU.z * u + detV.z * v);

	// detCenter to grid cx
	float margin1 = sqrtf((grid.cx - detCenter.x) * (grid.cx - detCenter.x) +
		(grid.cy - detCenter.y) * (grid.cy - detCenter.y) +
		(grid.cz - detCenter.z) * (grid.cz - detCenter.z));
	// radius of grid
	float margin2 = sqrtf(grid.nx * grid.dx * grid.nx * grid.dx +
		grid.ny * grid.dy * grid.ny * grid.dy +
		grid.nz * grid.dz * grid.nz * grid.dz) / 2;
	float margin = fmaxf(margin1, margin2) * 1.1f;
	src = make_float3(detCenter.x + invRayDir.x * margin + detU.x * u + detV.x * v,
		detCenter.y + invRayDir.y * margin + detU.y * u + detV.y * v,
		detCenter.z + invRayDir.z * margin + detU.z * u + detV.z * v);

}

// move source and dst near the grid to avoid precision problems in further ray tracing
__device__ __host__ void MoveSourceDstNearGrid(float3& src, float3&dst, const Grid grid)
{
	// twice the radius of grid
	float r = vecLength(make_float3(grid.nx * grid.dx, grid.ny * grid.dy, grid.nz * grid.dz));
	// dst to src direction
	float3 d = vecNormalize(vecMinus(dst, src));
	// distance from source to central plane
	float t0 = vecDot(make_float3(grid.cx - src.x, grid.cy - src.y, grid.cz - src.z), d);
	// distance from dst to central plane
	float t1 = vecDot(make_float3(dst.x - grid.cx, dst.y - grid.cy, dst.z - grid.cz), d);

	if (t0 > r)
	{
		src = make_float3(src.x + (t0 - r) * d.x, src.y + (t0 - r) * d.y, src.z + (t0 - r) * d.z);
	}

	if (t1 > r)
	{
		dst = make_float3(dst.x - (t1 - r) * d.x, dst.y - (t1 - r) * d.y, dst.z - (t1 - r) * d.z);
	}
}

__global__ void SiddonParallelProjectionKernel(float* pPrj, const float* pImg,
	const float3* pDetCenter, const float3* pDetU, const float3* pDetV, const float3* pInvRayDir,
	int nview, int nc, const Detector det, const Grid grid)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockDim.z * blockIdx.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;

	float3 dst, src;
	GetSrcDstForParallel(src, dst, u, v, pDetCenter[iview], pDetU[iview], pDetV[iview], pInvRayDir[iview], grid);
	MoveSourceDstNearGrid(src, dst, grid);

	SiddonRayTracing(pPrj + iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc,
				pImg, src, dst, nc, grid);

}

__global__ void SiddonParallelBackprojectionKernel(float* pImg, const float* pPrj,
		const float3* pDetCenter, const float3* pDetU, const float3* pDetV, const float3* pInvRayDir,
		int nview, int nc, const Detector det, const Grid grid)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockDim.z * blockIdx.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;

	float3 dst, src;
	GetSrcDstForParallel(src, dst, u, v, pDetCenter[iview], pDetU[iview], pDetV[iview], pInvRayDir[iview], grid);
	MoveSourceDstNearGrid(src, dst, grid);

	SiddonRayTracingTransposeAtomicAdd(pImg, pPrj[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc],
			src, dst, nc, grid);

}

void SiddonParallel::ProjectionParallel(const float* pcuImg, float* pcuPrj, const float* pcuDetCenter,
		const float* pcuDetU, const float* pcuDetV, const float* pcuInvRayDir)
{
	dim3 threads(32, 1, 32);
	dim3 blocks(ceilf(nu / (float)threads.x), nview, ceilf(nv / (float)threads.z));

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			SiddonParallelProjectionKernel<<<blocks, threads, 0, m_stream>>>(
					pcuPrj + ib * nu * nview * nv * nChannels + ic,
					pcuImg + ib * nx * ny * nz * nChannels + ic,
					(const float3*)pcuDetCenter, (const float3*)pcuDetU,
					(const float3*)pcuDetV, (const float3*)pcuInvRayDir,
					nview, nChannels, MakeDetector(nu, nv, du, dv, off_u, off_v),
					MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz));
			cudaDeviceSynchronize();
		}
	}

}

void SiddonParallel::BackprojectionParallel(float* pcuImg, const float* pcuPrj, const float* pcuDetCenter,
			const float* pcuDetU, const float* pcuDetV, const float* pcuInvRayDir)
{
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), nz);

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			SiddonParallelBackprojectionKernel<<<blocks, threads, 0, m_stream>>>(
					pcuImg + ib * nx * ny * nz * nChannels + ic,
					pcuPrj + ib * nu * nview * nv * nChannels + ic,
					(const float3*)pcuDetCenter, (const float3*)pcuDetU,
					(const float3*)pcuDetV, (const float3*)pcuInvRayDir,
					nview, nChannels, MakeDetector(nu, nv, du, dv, off_u, off_v),
					MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz));
			cudaDeviceSynchronize();
		}
	}
}

extern "C" void cSiddonParallelProjection(float* prj, const float* img,
		const float* detCenter, const float* detU, const float* detV, const float* invRayDir,
		int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float* pcuDetCenter = NULL;
	float* pcuDetU = NULL;
	float* pcuDetV = NULL;
	float* pcuInvRayDir = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nu * nv * nview * nBatches * nChannels))
		{
			throw ("pcuPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nx * ny * nz * nBatches * nChannels))
		{
			throw ("pcuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetCenter, sizeof(float3) * nview))
		{
			throw ("pcuDetCenter allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetU, sizeof(float3) * nview))
		{
			throw ("pcuDetU allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetV, sizeof(float3) * nview))
		{
			throw ("pcuDetV allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuInvRayDir, sizeof(float3) * nview))
		{
			throw ("pcuInvRayDir allocation failed");
		}
	}
	catch (exception& e)
	{
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
		if (pcuDetU != NULL) cudaFree(pcuDetU);
		if (pcuDetV != NULL) cudaFree(pcuDetV);
		if (pcuInvRayDir != NULL) cudaFree(pcuInvRayDir);

		ostringstream oss;
		oss << "cSiddonParallelProjection() failed: " << e.what()
				<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw(oss.str().c_str());
	}

	cudaMemcpy(pcuImg, img, sizeof(float) * nx * ny * nz * nBatches * nChannels, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuDetU, detU, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuDetV, detV, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuInvRayDir, invRayDir, sizeof(float3) * nview, cudaMemcpyHostToDevice);

	SiddonParallel projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nview, nv, du, dv, off_u, off_v, 0, 0, 2);

	projector.ProjectionParallel(pcuImg, pcuPrj, pcuDetCenter, pcuDetU, pcuDetV, pcuInvRayDir);
	cudaMemcpy(prj, pcuPrj, sizeof(float) * nu * nv * nview * nBatches * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuPrj);
	cudaFree(pcuImg);
	cudaFree(pcuDetCenter);
	cudaFree(pcuDetU);
	cudaFree(pcuDetV);
	cudaFree(pcuInvRayDir);

}

extern "C" void cSiddonParallelBackprojection(float* img, const float* prj,
		const float* detCenter, const float* detU, const float* detV, const float* invRayDir,
		int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float* pcuDetCenter = NULL;
	float* pcuDetU = NULL;
	float* pcuDetV = NULL;
	float* pcuInvRayDir = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nu * nv * nview * nBatches * nChannels))
		{
			throw ("pcuPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nx * ny * nz * nBatches * nChannels))
		{
			throw ("pcuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetCenter, sizeof(float3) * nview))
		{
			throw ("pcuDetCenter allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetU, sizeof(float3) * nview))
		{
			throw ("pcuDetU allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetV, sizeof(float3) * nview))
		{
			throw ("pcuDetV allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuInvRayDir, sizeof(float3) * nview))
		{
			throw ("pcuInvRayDir allocation failed");
		}
	}
	catch (exception& e)
	{
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
		if (pcuDetU != NULL) cudaFree(pcuDetU);
		if (pcuDetV != NULL) cudaFree(pcuDetV);
		if (pcuInvRayDir != NULL) cudaFree(pcuInvRayDir);

		ostringstream oss;
		oss << "cSiddonParallelProjection() failed: " << e.what()
				<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw(oss.str().c_str());
	}

	cudaMemcpy(pcuPrj, prj, sizeof(float) * nu * nv * nview * nBatches * nChannels, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuDetU, detU, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuDetV, detV, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuInvRayDir, invRayDir, sizeof(float3) * nview, cudaMemcpyHostToDevice);

	SiddonParallel projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nview, nv, du, dv, off_u, off_v, 0, 0, 2);

	projector.BackprojectionParallel(pcuImg, pcuPrj, pcuDetCenter, pcuDetU, pcuDetV, pcuInvRayDir);
	cudaMemcpy(img, pcuImg, sizeof(float) * nx * ny * nz * nBatches * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuPrj);
	cudaFree(pcuImg);
	cudaFree(pcuDetCenter);
	cudaFree(pcuDetU);
	cudaFree(pcuDetV);
	cudaFree(pcuInvRayDir);

}
