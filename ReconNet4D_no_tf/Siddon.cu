#include "Projector.h"
#include "Siddon.h"
#include "cudaMath.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

#define eps 1e-6f

// image coordinate: the origin is at the lower left corner of the first pixel
__device__ float3 PhysicsToImg(float3 pt, const Grid grid)
{
	pt.x = (pt.x - grid.cx) / grid.dx + grid.nx / 2.0f;
	pt.y = (pt.y - grid.cy) / grid.dy + grid.ny / 2.0f;
	pt.z = (pt.z - grid.cz) / grid.dz + grid.nz / 2.0f;

	return pt;
}

__device__ bool InboundAlpha(float& alpha0v, float& alphaNv, float dDstSrcv, float srcv, int gridnv)
{
	if (fabsf(dDstSrcv) > eps)
	{
		// non-parallel along x
		alpha0v = (0 - srcv) / dDstSrcv;
		alphaNv = (gridnv - srcv) / dDstSrcv;
	}
	else
	{
		// parallel along x
		if (srcv < 0 || srcv > gridnv)
		{
			// no intersection
			return false;
		}
	}

	return true;
}

// the first intersection of the ray with the grid, given separately for x, y, and z
__device__ int InboundFirstVoxel(float pt0v, float dDstSrcv, int gridnv)
{
	int ngv = (int)pt0v;
	if (fabs(dDstSrcv) > eps)
	{
		if (ngv < 0)
		{
			ngv = 0;
		}
		else if (ngv >= gridnv)
		{
			ngv = gridnv - 1;
		}
	}

	return ngv;
}

// the second intersection of the raw with the grid, gien seperately for x, y, and z
__device__ float OutboundFirstVoxel(int ngv, float dDstSrcv)
{
	if (dDstSrcv > eps)
	{
		return (float)(ngv + 1);
	}
	else if (dDstSrcv < eps)
	{
		return (float)(ngv);
	}
	else
	{
		return (float)(ngv);
	}
}

// get the alpha of the second intersection and the dAlpha along each direction
__device__ void GetAlpha(float& alphav, float& dAlphav, float pt1v, float srcv, float dDstSrcv)
{
	if (fabsf(dDstSrcv) > eps)
	{
		alphav = (pt1v - srcv) / dDstSrcv;
		dAlphav = 1 / fabsf(dDstSrcv);
	}
}

__device__ SiddonTracingVars InitializeSiddon(float3 src, float3 dst, const Grid grid)
{
	SiddonTracingVars var;

	float dist = sqrtf((src.x - dst.x) * (src.x - dst.x) +
			(src.y - dst.y) * (src.y - dst.y) +
			(src.z - dst.z) * (src.z - dst.z));

	src = PhysicsToImg(src, grid);
	dst = PhysicsToImg(dst, grid);

	float3 dDstSrc = make_float3(dst.x - src.x, dst.y - src.y, dst.z - src.z);

	// intersection between ray and grid
	float3 a0 = make_float3(1.0f, 1.0f, 1.0f);
	float3 a1 = make_float3(0.0f, 0.0f, 0.0f);
	if (!InboundAlpha(a0.x, a1.x, dDstSrc.x, src.x, grid.nx) ||
		!InboundAlpha(a0.y, a1.y, dDstSrc.y, src.y, grid.ny) ||
		!InboundAlpha(a0.z, a1.z, dDstSrc.z, src.z, grid.nz))
	{
		var.alpha.x = -1;
		return var;
	}

	// entry and exit points
	float amin = fmaxf(0.f, fmaxf(fminf(a0.x, a1.x), fmaxf(fminf(a0.y, a1.y), fminf(a0.z, a1.z))));	// entry
	float amax = fminf(1.f, fminf(fmaxf(a0.x, a1.x), fminf(fmaxf(a0.y, a1.y), fmaxf(a0.z, a1.z))));	// exit

	if (amax <= amin)
	{
		// no intersection
		var.alpha.x = -1;
		return var;
	}

	// entry point
	float3 pt0 = make_float3(src.x + amin * dDstSrc.x, src.y + amin * dDstSrc.y, src.z + amin * dDstSrc.z);

	// first intersection voxel
	int3 ng0 = make_int3(InboundFirstVoxel(pt0.x, dDstSrc.x, grid.nx),
		InboundFirstVoxel(pt0.y, dDstSrc.y, grid.ny),
		InboundFirstVoxel(pt0.z, dDstSrc.z, grid.nz));

	// exiting point from first voxel
	float3 pt1 = make_float3(OutboundFirstVoxel(ng0.x, dDstSrc.x),
			OutboundFirstVoxel(ng0.y, dDstSrc.y),
			OutboundFirstVoxel(ng0.z, dDstSrc.z));

	// the alpha of the exiting point and step size along each direction
	float3 alpha = make_float3(2.0f, 2.0f, 2.0f);	// the max value of alpha is 1, so if alpha is not set in any direction then it will be skipped
	float3 dAlpha = make_float3(0.0f, 0.0f, 0.0f);
	GetAlpha(alpha.x, dAlpha.x, pt1.x, src.x, dDstSrc.x);
	GetAlpha(alpha.y, dAlpha.y, pt1.y, src.y, dDstSrc.y);
	GetAlpha(alpha.z, dAlpha.z, pt1.z, src.z, dDstSrc.z);

	var.alpha = make_float3(alpha.x * dist, alpha.y * dist, alpha.z * dist);
	var.dAlpha = make_float3(dAlpha.x * dist, dAlpha.y * dist, dAlpha.z * dist);
	var.ng = ng0;
	var.isPositive = make_int3((dDstSrc.x > 0) ? 1 : 0, (dDstSrc.y > 0) ? 1 : 0, (dDstSrc.z > 0) ? 1 : 0);
	var.alphaNow = var.alphaPrev = amin * dist;
	var.rayLength = dist * (amax - amin);

	return var;
}

__device__ float SiddonRayTracing(float* pPrj, const float* pImg, float3 src, float3 dst, int nc, const Grid grid)
{
	SiddonTracingVars var = InitializeSiddon(src, dst, grid);

	if (var.alpha.x < -0.5 || var.ng.x < 0 || var.ng.x >= grid.nx
			|| var.ng.y < 0 || var.ng.y >= grid.ny
			|| var.ng.z < 0 || var.ng.z >= grid.nz)
	{
		// no intersections
		return 0;
	}

	int move = 0;
	bool isTracing = true;
	float val = 0;
	int nzc = grid.nz * nc;
	int nyzc = grid.ny * nzc;
	pImg += var.ng.x * nyzc + var.ng.y * nzc + var.ng.z * nc;

	while (isTracing)
	{
		// each iteration find the direction of alpha nearest to the src,
		// set it to alphaNow then move it to the next intersection with grid along that direction
		if (var.alpha.x < var.alpha.y && var.alpha.x < var.alpha.z)
		{
			var.alphaNow = var.alpha.x;
			var.alpha.x += var.dAlpha.x;
			if (var.isPositive.x == 1)
			{
				var.ng.x++;
				move = nyzc;
				if (var.ng.x >= grid.nx) isTracing = false;
			}
			else
			{
				var.ng.x--;
				move = -nyzc;
				if (var.ng.x < 0) isTracing = false;
			}
		}
		else if (var.alpha.y < var.alpha.z)
		{
			var.alphaNow = var.alpha.y;
			var.alpha.y += var.dAlpha.y;
			if (var.isPositive.y == 1)
			{
				var.ng.y++;
				move = nzc;
				if (var.ng.y >= grid.ny) isTracing = false;
			}
			else
			{
				var.ng.y--;
				move = -nzc;
				if (var.ng.y < 0) isTracing = false;
			}
		}
		else
		{
			var.alphaNow = var.alpha.z;
			var.alpha.z += var.dAlpha.z;
			if (var.isPositive.z == 1)
			{
				var.ng.z++;
				move = nc;
				if (var.ng.z >= grid.nz) isTracing = false;
			}
			else
			{
				var.ng.z--;
				move = -nc;
				if (var.ng.z < 0) isTracing = false;
			}
		}

		val += (*pImg) * (var.alphaNow - var.alphaPrev);
		var.alphaPrev = var.alphaNow;
		pImg += move;
	}

	*pPrj += val;

	return var.rayLength;
}

__global__ void SiddonFanProjectionKernel(float* pPrj, const float* pImg,
	const float* pDeg, int nview, int nc, const Grid grid, const Detector det,
	float dsd, float dso)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iview >= nview || iv >= det.nv)
	{
		return;
	}

	float cosDeg = __cosf(pDeg[iview]);
	float sinDeg = __sinf(pDeg[iview]);
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// src, dst points, and convert to image coordinate
	float3 src = make_float3(dso * sinDeg, -dso * cosDeg, z);
	float3 dstRel = make_float3(dsd * sin_a, -dso + dsd * cos_a, z);
	float3 dst = make_float3(dstRel.x * cosDeg - dstRel.y * sinDeg, dstRel.x * sinDeg + dstRel.y * cosDeg, z);

	SiddonRayTracing(pPrj + iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc,
			pImg, src, dst, nc, grid);

}

__device__ float SiddonRayTracingTransposeAtomicAdd(float* pImg, float val, float3 src, float3 dst,
		int nc, const Grid grid)
{
	SiddonTracingVars var = InitializeSiddon(src, dst, grid);

	if (var.alpha.x < -0.5 || var.ng.x < 0 || var.ng.x >= grid.nx
			|| var.ng.y < 0 || var.ng.y >= grid.ny
			|| var.ng.z < 0 || var.ng.z >= grid.nz)
	{
		// no intersections
		return 0;
	}

	int move = 0;
	bool isTracing = true;
	int nzc = grid.nz * nc;
	int nyzc = grid.ny * nzc;
	pImg += var.ng.x * nyzc + var.ng.y * nzc + var.ng.z * nc;

	while (isTracing)
	{
		// each iteration find the direction of alpha nearest to the src,
		// set it to alphaNow then move it to the next intersection with grid along that direction
		if (var.alpha.x < var.alpha.y && var.alpha.x < var.alpha.z)
		{
			var.alphaNow = var.alpha.x;
			var.alpha.x += var.dAlpha.x;
			if (var.isPositive.x == 1)
			{
				var.ng.x++;
				move = nyzc;
				if (var.ng.x >= grid.nx) isTracing = false;
			}
			else
			{
				var.ng.x--;
				move = -nyzc;
				if (var.ng.x < 0) isTracing = false;
			}
		}
		else if (var.alpha.y < var.alpha.z)
		{
			var.alphaNow = var.alpha.y;
			var.alpha.y += var.dAlpha.y;
			if (var.isPositive.y == 1)
			{
				var.ng.y++;
				move = nzc;
				if (var.ng.y >= grid.ny) isTracing = false;
			}
			else
			{
				var.ng.y--;
				move = -nzc;
				if (var.ng.y < 0) isTracing = false;
			}
		}
		else
		{
			var.alphaNow = var.alpha.z;
			var.alpha.z += var.dAlpha.z;
			if (var.isPositive.z == 1)
			{
				var.ng.z++;
				move = nc;
				if (var.ng.z >= grid.nz) isTracing = false;
			}
			else
			{
				var.ng.z--;
				move = -nc;
				if (var.ng.z < 0) isTracing = false;
			}
		}

		atomicAdd(pImg, val * (var.alphaNow - var.alphaPrev));
		var.alphaPrev = var.alphaNow;
		pImg += move;
	}

	return var.rayLength;
}


__global__ void SiddonFanBackprojectionKernel(float* pImg, const float* pPrj,
	const float* pDeg, int nview, int nc, const Grid grid, const Detector det,
	float dsd, float dso)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iview >= nview || iv >= det.nv)
	{
		return;
	}

	float cosDeg = __cosf(pDeg[iview]);
	float sinDeg = __sinf(pDeg[iview]);
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// src, dst points, and convert to image coordinate
	float3 src = make_float3(dso * sinDeg, -dso * cosDeg, z);
	float3 dstRel = make_float3(dsd * sin_a, -dso + dsd * cos_a, z);
	float3 dst = make_float3(dstRel.x * cosDeg - dstRel.y * sinDeg, dstRel.x * sinDeg + dstRel.y * cosDeg, z);

	SiddonRayTracingTransposeAtomicAdd(pImg, pPrj[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc],
			src, dst, nc, grid);

}

void SiddonFan::Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg)
{
	dim3 threads, blocks;
	GetThreadsForXY(threads, blocks, nu, nview, nv);

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			SiddonFanProjectionKernel<<<blocks, threads, 0, m_stream>>>(
					pcuPrj + ib * nu * nview * nv * nChannels + ic,
					pcuImg + ib * nx * ny * nz * nChannels + ic, pcuDeg, nview, nChannels,
					MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz),
					MakeDetector(nu, nv, du, dv, off_u, off_v), dsd, dso);
			cudaDeviceSynchronize();
		}
	}

}

void SiddonFan::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	dim3 threads, blocks;
	GetThreadsForXY(threads, blocks, nu, nview, nv);

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			SiddonFanBackprojectionKernel<<<blocks, threads, 0, m_stream>>>(
					pcuImg + ib * nx * ny * nz * nChannels + ic,
					pcuPrj + ib * nu * nview * nv * nChannels + ic, pcuDeg, nview, nChannels,
					MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz),
					MakeDetector(nu, nv, du, dv, off_u, off_v), dsd, dso);
			cudaDeviceSynchronize();
		}
	}

}

extern "C" void cSiddonFanProjection(float* prj, const float* img, const float* deg,
		int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, int typeProjector)
{
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

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuImg, img, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyHostToDevice);
		cudaMemset(pcuPrj, 0, sizeof(float) * nBatches * nu * nview * nv * nChannels);
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuDeg != NULL) cudaFree(pcuDeg);

		ostringstream oss;
		oss << "cSiddonFanProjection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	SiddonFan projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeProjector);

	projector.Projection(pcuImg, pcuPrj, pcuDeg);
	cudaMemcpy(prj, pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuDeg);


}


extern "C" void cSiddonFanBackprojection(float* img, const float* prj, const float* deg,
		int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, int typeProjector)
{
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

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuPrj, prj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels);
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuDeg != NULL) cudaFree(pcuDeg);

		ostringstream oss;
		oss << "cSiddonFanBackprojection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	SiddonFan projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeProjector);

	projector.Backprojection(pcuImg, pcuPrj, pcuDeg);
	cudaMemcpy(img, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuDeg);


}



