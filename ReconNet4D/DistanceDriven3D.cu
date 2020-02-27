#include "DistanceDriven.h"
#include "cudaMath.h"
#include "Siddon.h"

#include <stdexcept>
#include <exception>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

// DD projection branchless version

// origin of image coordinate is the corner of the first pixel, the same with texture coordinate.
__device__ static float3 PhysicsToImg(float3 pt, const Grid grid)
{
	pt.x = (pt.x - grid.cx) / grid.dx + grid.nx / 2.0f;
	pt.y = (pt.y - grid.cy) / grid.dy + grid.ny / 2.0f;
	pt.z = (pt.z - grid.cz) / grid.dz + grid.nz / 2.0f;

	return pt;
}

__device__ __host__ static float3 GetDstForCone(float u, float v,
		const float3& detCenter, const float3& detU, const float3& detV)
{
	return make_float3(detCenter.x + detU.x * u + detV.x * v,
		detCenter.y + detU.y * u + detV.y * v,
		detCenter.z + detU.z * u + detV.z * v);

}

// the dst is realigned to make channel a higher dimension than x, y, z for later texture binding
// dst and src should point to different address unless nc is 1
// use double precision for dst because potentially a lot of pixels hhhas to be added
__global__ void AccumulateXYAlongXKernel(double* dst, const float* src, int nx, int ny, int nz, int nc)
{
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (iy >= ny || iz >= nz)
	{
		return;
	}

	dst += (iy + 1) * nz + iz;	// skip iy == 0, which should always be zero in dst
	src += iy * nz * nc + iz * nc;
	dst[0] = 0;
	for (int ix = 0; ix < nx; ix++)
	{
		dst[(ix + 1) * (ny + 1) * nz] = dst[ix * (ny + 1) * nz] + src[ix * ny * nz * nc];
	}
}

// this kernel should be called right after AccumulateXYAlongXKernel to integrate along y axis,
// no dimension switch since channel should already be switched to higher dimension in the previous call
__global__ void AccumulateXYAlongYKernel(double* acc, int nx, int ny, int nz, int nc)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (ix >= nx || iz >= nz)
	{
		return;
	}

	acc += (ix + 1) * (ny + 1) * nz + iz;	// skip ix == 0, which should always be zero in dst
	for (int iy = 0; iy < ny; iy++)
	{
		acc[(iy + 1) * nz] = acc[(iy + 1) * nz] + acc[iy * nz];
	}
}

__device__ static float Clamp_float(float x, float start, float end)
{
	if (x < start)
	{
		return start;
	}
	else if (x >= end)
	{
		return end - 1;
	}
	else
	{
		return x;
	}
}


// clamp x to range [start, end)
__device__ static int Clamp(int x, int start, int end)
{
	if (x < start)
	{
		return start;
	}
	else if (x >= end)
	{
		return end - 1;
	}
	else
	{
		return x;
	}
}

// 2d interpolation
__device__ static double InterpolateXY(const double* acc, float x, float y, int iz, int nx, int ny, int nz)
{
	x = Clamp_float(x, 0, nx);
	y = Clamp_float(y, 0, ny);

	int ix = int(x);
	int iy = int(y);
	int ix1 = ix + 1;
	int iy1 = iy + 1;

//	ix = Clamp(ix, 0, nx);
//	iy = Clamp(iy, 0, ny);
	ix1 = Clamp(ix1, 0, nx);
	iy1 = Clamp(iy1, 0, ny);


	double wx = (x - ix);
	double wy = (y - iy);

	return double(acc[ix * ny * nz + iy * nz + iz] * (1 - wx) * (1 - wy) +
			acc[ix1 * ny * nz + iy * nz + iz] * wx * (1 - wy) +
			acc[ix * ny * nz + iy1 * nz + iz] * (1 - wx) * wy +
			acc[ix1 * ny * nz + iy1 * nz + iz] * wx * wy);


}

// iviews - the list of iview where DDFP is performed on the XY plane
// nValidViews - length of iviews
__global__ void DDFPConeKernelXY(float* pPrjs, const double* acc,
		const int* iviews, int nValidViews, int nview, int nc,
		const float3* pDetCenter, const float3* pDetU, const float3* pDetV, const float3* pSrc,
		const Grid grid, const Detector det)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int ind = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || ind >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[ind];

	// coordinates of the center point of each edge of the detector's unit
	// dstx1, dstx2 are the edges at +- u along x axis
	// dsty1, dsty2 are the edges at +- v along y axis
	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;
	float3 dstx1 = GetDstForCone(u - det.du / 2, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	float3 dstx2 = GetDstForCone(u + det.du / 2, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	float3 dsty1 = GetDstForCone(u, v - det.dv / 2, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	float3 dsty2 = GetDstForCone(u, v + det.dv / 2, pDetCenter[iview], pDetU[iview], pDetV[iview]);

	float3 src = pSrc[iview];

	// convert to image coordinate
	src = PhysicsToImg(src, grid);
	dstx1 = PhysicsToImg(dstx1, grid);
	dstx2 = PhysicsToImg(dstx2, grid);
	dsty1 = PhysicsToImg(dsty1, grid);
	dsty2 = PhysicsToImg(dsty2, grid);

	// make sure dstx1.x < dstx2.x
	if (dstx1.x > dstx2.x)
	{
		float3 tmp = dstx1;
		dstx1 = dstx2;
		dstx2 = tmp;
	}

	// make sure dsty1.y < dsty2.y
	if (dsty1.y > dsty2.y)
	{
		float3 tmp = dsty1;
		dsty1 = dsty2;
		dsty2 = tmp;
	}

	float val = 0;
	float rx1 = (dstx1.x - src.x) / (dstx1.z - src.z);
	float rx2 = (dstx2.x - src.x) / (dstx2.z - src.z);
	float ry1 = (dsty1.y - src.y) / (dsty1.z - src.z);
	float ry2 = (dsty2.y - src.y) / (dsty2.z - src.z);

	// calculate intersection with each xy plane at different z
	for (int iz = 0; iz < grid.nz; iz++)
	{
		float x1 = src.x + rx1 * (iz - src.z);
		float x2 = src.x + rx2 * (iz - src.z);
		float y1 = src.y + ry1 * (iz - src.z);
		float y2 = src.y + ry2 * (iz - src.z);

//		val += InterpolateXY(acc, x2, y2, iz, grid.nx+1, grid.ny+1, grid.nz);

		val += (InterpolateXY(acc, x2, y2, iz, grid.nx+1, grid.ny+1, grid.nz)
				+ InterpolateXY(acc, x1, y1, iz, grid.nx+1, grid.ny+1, grid.nz)
				- InterpolateXY(acc, x2, y1, iz, grid.nx+1, grid.ny+1, grid.nz)
				- InterpolateXY(acc, x1, y2, iz, grid.nx+1, grid.ny+1, grid.nz)) / ((x2 - x1) * (y2 - y1));

		// (0.5, 0.5) of texAcc is the integral at the border of image, since x1,x2,y1,y2 are coordinates on the image, so
		// an offset of +0.5 should be added when fetching the integral value
//		val += (tex3D<float>(texAcc, iz + 0.5f, y2 + 0.5f, x2 + 0.5f) + tex3D<float>(texAcc, iz + 0.5f, y1 + 0.5f, x1 + 0.5f)
//				- tex3D<float>(texAcc, iz + 0.5f, y1 + 0.5f, x2 + 0.5f) - tex3D<float>(texAcc, iz + 0.5f, y2 + 0.5f, x1 + 0.5f))
//						/ ((x2 - x1) * (y2 - y1));

	}

	// normalize by length
	// use physics coordinate
	float3 dst = GetDstForCone(u, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	src = pSrc[iview];
	val *= grid.dz / fabsf((src.z - dst.z)) * sqrtf((src.z-dst.z)*(src.z-dst.z) + (src.y-dst.y)*(src.y-dst.y) + (src.x-dst.x)*(src.x-dst.x));

	pPrjs[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc] = val;

}

// no textures, use double-precision software interpolation
void DistanceDrivenTomo::ProjectionTomo(const float* pcuImg, float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc)
{
	double* pAcc = NULL;
	int* cuIviews = NULL;

	float3* pcuDetU = NULL;
	float3* pcuDetV = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pAcc, sizeof(double) * (nx+1) * (ny+1) * nz * nBatches * nChannels))
		{
			throw runtime_error("pAcc allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDetU, sizeof(float3) * nview))
		{
			throw runtime_error("pDetU allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDetV, sizeof(float3) * nview))
		{
			throw runtime_error("pDetV allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&cuIviews, sizeof(int) * nview))
		{
			throw runtime_error("cuIviews allocation failed");
		}

	}
	catch (exception &e)
	{
		if (pAcc != NULL) cudaFree(pAcc);
		if (cuIviews != NULL) cudaFree(cuIviews);
		if (pcuDetU != NULL) cudaFree(pcuDetU);
		if (pcuDetV != NULL) cudaFree(pcuDetV);

		ostringstream oss;
		oss << "DistanceDrivenTomo::ProjectionTomo Error: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

	// cuIviews should contain all angles
	int* iviews = new int [nview];
	for (int i = 0; i < nview; i++)
	{
		iviews[i] = i;
	}
	cudaMemcpy(cuIviews, iviews, sizeof(int) * nview, cudaMemcpyHostToDevice);
	delete [] iviews;

	// pcuDetU should contain all (1,0,0)
	// pcuDetV should contain all (0,1,0)
	float3* pDetU = new float3 [nview];
	for (int i = 0; i < nview; i++)
	{
		pDetU[i] = make_float3(1, 0, 0);
	}
	cudaMemcpy(pcuDetU, pDetU, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	delete [] pDetU;

	float3* pDetV = new float3 [nview];
	for (int i = 0; i < nview; i++)
	{
		pDetV[i] = make_float3(0, 1, 0);
	}
	cudaMemcpy(pcuDetV, pDetV, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	delete [] pDetV;

	Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
	Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

	// step 1: calculate accumulated images
	dim3 threadX(1,16,16);
	dim3 blockX(1, ceilf(ny / 16.0f), ceilf(nz / 16.0f));
	dim3 threadY(16,1,16);
	dim3 blockY(ceilf(nx / 16.0f), 1, ceilf(nz / 16.0f));
	cudaMemset(pAcc, 0, sizeof(double) * (nx + 1) * (ny + 1) * nz * nBatches * nChannels);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			// pAcc has the dimension in order (batch, channel, x, y, z) so that can be copied to textures
			AccumulateXYAlongXKernel<<<blockX, threadX>>>(pAcc + (ib * nChannels + ic) * (nx + 1) * (ny + 1) * nz,
					pcuImg + ib * nx * ny * nz * nChannels + ic , nx, ny, nz, nChannels);
			cudaDeviceSynchronize();
			AccumulateXYAlongYKernel<<<blockY, threadY>>>(pAcc + (ib * nChannels + ic) * (nx + 1) * (ny + 1) * nz,
					nx, ny, nz, nChannels);
		}
	}
	cudaDeviceSynchronize();

	// step 2: interpolation
	dim3 threads(16, 1, 16);
	dim3 blocks(ceilf(nu / 16.f), nview, ceilf(nv / 16.f));
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			DDFPConeKernelXY<<<blocks, threads>>>(pcuPrj + ib * nu * nview * nv * nChannels + ic,
					pAcc + (ib * nChannels + ic) * (nx + 1) * (ny + 1) * nz, cuIviews, nview, nview, nChannels,
					(const float3*)pcuDetCenter, pcuDetU, pcuDetV, (const float3*)pcuSrc, grid, det);

			cudaDeviceSynchronize();
		}
	}

	if (pAcc != NULL) cudaFree(pAcc);
	if (cuIviews != NULL) cudaFree(cuIviews);
	if (pcuDetU != NULL) cudaFree(pcuDetU);
	if (pcuDetV != NULL) cudaFree(pcuDetV);
}

// C interface
extern "C" void cDistanceDrivenTomoProjection(float* prj, const float* img,
		const float* detCenter, const float* src,
		int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float* pcuDetCenter = NULL;
	float* pcuSrc = NULL;

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
		if (cudaSuccess != cudaMalloc(&pcuSrc, sizeof(float3) * nview))
		{
			throw ("pcuSrc allocation failed");
		}
	}
	catch (exception& e)
	{
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
		if (pcuSrc != NULL) cudaFree(pcuSrc);

		ostringstream oss;
		oss << "cDistanceDrivenTomoProjection() failed: " << e.what()
				<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw(oss.str().c_str());
	}

	cudaMemcpy(pcuImg, img, sizeof(float) * nx * ny * nz * nBatches * nChannels, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuSrc, src, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemset(pcuPrj, 0, sizeof(float) * nu * nview * nv * nBatches * nChannels);

	DistanceDrivenTomo projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nview, nv, du, dv, off_u, off_v, 0, 0, 0);

	projector.ProjectionTomo(pcuImg, pcuPrj, pcuDetCenter, pcuSrc);
	cudaMemcpy(prj, pcuPrj, sizeof(float) * nu * nv * nview * nBatches * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuPrj);
	cudaFree(pcuImg);
	cudaFree(pcuDetCenter);
	cudaFree(pcuSrc);

}

__global__ void PreweightBPCartKernelXY(float* pPrjs,
		const int* iviews, int nValidViews, int nview, int nc,
		const float3* pDetCenter, const float3* pSrc, float dz, const Detector det)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iviewInd = blockIdx.y * blockDim.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iviewInd >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[iviewInd];

	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;
	float3 dst = GetDstForCone(u, v, pDetCenter[iview], make_float3(1,0,0), make_float3(0,1,0));
	float3 src = pSrc[iview];

	pPrjs[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc] *=
			dz / fabsf((src.z - dst.z)) * sqrtf((src.z-dst.z)*(src.z-dst.z) + (src.y-dst.y)*(src.y-dst.y) + (src.x-dst.x)*(src.x-dst.x));;
}

// dst has the dimension (batch, channel, nu+1, nview, nv+1)
// src has the dimension (batch, nu, nview, nv, channel)
// dst and src should point to different address unless nc is 1
// use double precision for dst because potentially a lot of pixels hhhas to be added
__global__ void AccumulateUVAlongUKernel(double* dst, const float* src, int nu, int nview, int nv, int nc)
{
	int iview = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockDim.z * blockIdx.z + threadIdx.z;

	if (iview >= nview || iv >= nv)
	{
		return;
	}

	dst += iview * (nv + 1) + iv + 1;	// skip iv == 0, which should always be zero in dst
	src += iview * nv * nc + iv * nc;
	dst[0] = 0;
	for (int iu = 0; iu < nu; iu++)
	{
		dst[(iu + 1) * nview * (nv + 1)] = dst[iu * nview * (nv + 1)] + src[iu * nview * nv * nc];
	}
}

// this kernel should be called right after AccumulateUVAlongUKernel to integrate along y axis,
// no dimension switch since channel should already be switched to higher dimension in the previous call
// acc has the dimension (batch, channel, nu+1, nview, nv+1)
__global__ void AccumulateUVAlongVKernel(double* acc, int nu, int nview, int nv)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.y * blockIdx.y + threadIdx.y;

	if (iu >= nu || iview >= nview)
	{
		return;
	}

	acc += (iu + 1) * nview * (nv + 1) + iview * (nv + 1);	// skip iu == 0, which should always be zero in dst
	for (int iv = 0; iv < nv; iv++)
	{
		acc[iv + 1] = acc[iv + 1] + acc[iv];
	}
}

// for fast calculation, assumed detU = (1,0,0) and detV = (0,1,0)
// directly convert to image coordinate
__device__ static float2 ProjectConeToDetCart(float3 pt, float3 detCenter, float3 src, const Detector& det)
{
	float r = (detCenter.z - src.z) / (pt.z - src.z);
	float u = src.x - detCenter.x + r * (pt.x - src.x);
	float v = src.y - detCenter.y + r * (pt.y - src.y);

	u = u / det.du + det.off_u + (det.nu - 1) / 2.0f;
	v = v / det.dv + det.off_v + (det.nv - 1) / 2.0f;

	return make_float2(u, v);

}

// 2d interpolation
// acc has dimension (nu, nview, nv)
__device__ static double InterpolateUV(const double* acc, float u, float v, int iview, int nu, int nv, int nview)
{
	u = Clamp_float(u, 0, nu);
	v = Clamp_float(v, 0, nv);

	int iu = int(u);
	int iv = int(v);
	int iu1 = iu + 1;
	int iv1 = iv + 1;

//	iu = Clamp(iu, 0, nu);
//	iv = Clamp(iv, 0, nv);
	iu1 = Clamp(iu1, 0, nu);
	iv1 = Clamp(iv1, 0, nv);

	double wu = (u - iu);
	double wv = (v - iv);

	return double(acc[iu * nview * nv + iview * nv + iv] * (1 - wu) * (1 - wv) +
			acc[iu1 * nview * nv + iview * nv + iv] * wu * (1 - wv) +
			acc[iu * nview * nv + iview * nv + iv1] * (1 - wu) * wv +
			acc[iu1 * nview * nv + iview * nv + iv1] * wu * wv);


}

__global__ void DDBPConeCartKernelXY(float* pImg, const double* acc,
		const int* iviews, int nValidViews, int nview, int nc,
		const float3* pDetCenter, const float3* pSrc,
		const Grid grid, const Detector det)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= grid.nx || iy >= grid.ny || iz >= grid.nz)
	{
		return;
	}

	float x = (ix - (grid.nx - 1) / 2.0f) * grid.dx + grid.cx;
	float y = (iy - (grid.ny - 1) / 2.0f) * grid.dy + grid.cy;
	float z = (iz - (grid.nz - 1) / 2.0f) * grid.dz + grid.cz;

	float val = 0;
	for (int ind = 0; ind < nValidViews; ind++)
	{
		int iview = iviews[ind];
		float3 src = pSrc[iview];
		float3 detCenter = pDetCenter[iview];

		float u1 = ProjectConeToDetCart(make_float3(x - grid.dx / 2, y, z), detCenter, src, det).x;
		float u2 = ProjectConeToDetCart(make_float3(x + grid.dx / 2, y, z), detCenter, src, det).x;
		float v1 = ProjectConeToDetCart(make_float3(x, y - grid.dy / 2, z), detCenter, src, det).y;
		float v2 = ProjectConeToDetCart(make_float3(x, y + grid.dy / 2, z), detCenter, src, det).y;


		val += (InterpolateUV(acc, u2, v2, iview, det.nu + 1, det.nv + 1, nview)
				- InterpolateUV(acc, u2, v1, iview, det.nu + 1, det.nv + 1, nview)
				+ InterpolateUV(acc, u1, v1, iview, det.nu + 1, det.nv + 1, nview)
				- InterpolateUV(acc, u1, v2, iview, det.nu + 1, det.nv + 1, nview)) / ((u2 - u1) * (v2 - v1));

	}

	pImg[ix * grid.ny * grid.nz * nc + iy * grid.nz * nc + iz * nc] = val;

}

// no textures, use double-precision software interpolation
void DistanceDrivenTomo::BackprojectionTomo(float* pcuImg, const float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc)
{
	// the backprojection is constrained to Cartesian coordinate for simplification, hence no detU / detV needed
	float* pWeightedPrjs = NULL;
	double* pAcc = NULL;
	int* cuIviews = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pWeightedPrjs, sizeof(float) * nBatches * nChannels * nu * nview * nv))
		{
			throw runtime_error("pWeightedPrjs allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pAcc, sizeof(double) * (nu+1) * nview * (nv+1) * nBatches * nChannels))
		{
			throw runtime_error("pAcc allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&cuIviews, sizeof(int) * nview))
		{
			throw runtime_error("cuIviews allocation failed");
		}

	}
	catch (exception &e)
	{
		if (pWeightedPrjs != NULL) cudaFree(pWeightedPrjs);
		if (pAcc != NULL) cudaFree(pAcc);
		if (cuIviews != NULL) cudaFree(cuIviews);

		ostringstream oss;
		oss << "DistanceDrivenTomo::BackprojectionTomo Error: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

	// cuIviews should contain all angles
	int* iviews = new int [nview];
	for (int i = 0; i < nview; i++)
	{
		iviews[i] = i;
	}
	cudaMemcpy(cuIviews, iviews, sizeof(int) * nview, cudaMemcpyHostToDevice);
	delete [] iviews;

	Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
	Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

	// step 0: preweight the projections for ray intersection length
	dim3 threadUV(16, 1, 16);
	dim3 blockUV(ceilf(nu / 16.f), 1, ceilf(nv / 16.f));
	cudaMemcpy(pWeightedPrjs, pcuPrj, sizeof(float) * nBatches * nChannels * nu * nview * nv, cudaMemcpyDeviceToDevice);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			PreweightBPCartKernelXY<<<blockUV, threadUV>>>(pWeightedPrjs + ib * nu * nview * nv * nChannels + ic, cuIviews,
					nview, nview, nChannels, (const float3*)pcuDetCenter, (const float3*)pcuSrc, grid.dz, det);
		}
	}
	cudaDeviceSynchronize();

	// step 1: calculate accumulated projections
	dim3 threadU(1,4,32);
	dim3 blockU(1, ceilf(nview / 4.0f), ceilf(nv / 32.0f));
	dim3 threadV(32,4,1);
	dim3 blockV(ceilf(nu / 32.0f), ceilf(nview / 4.0f), 1);
	cudaMemset(pAcc, 0, sizeof(double) * (nu + 1) * nview * (nv + 1) * nBatches * nChannels);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			// pAcc has the dimension in order (batch, channel, x, y, z) so that can be copied to textures
			AccumulateUVAlongUKernel<<<blockU, threadU>>>(pAcc + (ib * nChannels + ic) * (nu + 1) * nview * (nv + 1),
					pWeightedPrjs + ib * nu * nview * nv * nChannels + ic , nu, nview, nv, nChannels);
			cudaDeviceSynchronize();
			AccumulateUVAlongVKernel<<<blockV, threadV>>>(pAcc + (ib * nChannels + ic) * (nu + 1) * nview * (nv + 1),
					nu, nview, nv);
		}
	}
	cudaDeviceSynchronize();

	// step 2: interpolation
	dim3 threads(16, 16, 1);
	dim3 blocks(ceilf(nx / 16.f), ceilf(ny / 16.f), nz);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			DDBPConeCartKernelXY<<<blocks, threads>>>(pcuImg + ib * nx * ny * nz * nChannels + ic,
					pAcc + (ib * nChannels + ic) * (nu + 1) * nview * (nv + 1), cuIviews, nview, nview, nChannels,
					(const float3*)pcuDetCenter, (const float3*)pcuSrc, grid, det);

			cudaDeviceSynchronize();
		}
	}

	if (pWeightedPrjs != NULL) cudaFree(pWeightedPrjs);
	if (pAcc != NULL) cudaFree(pAcc);
	if (cuIviews != NULL) cudaFree(cuIviews);
}

// C interface
extern "C" void cDistanceDrivenTomoBackprojection(float* img, const float* prj,
		const float* detCenter, const float* src,
		int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float* pcuDetCenter = NULL;
	float* pcuSrc = NULL;

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
		if (cudaSuccess != cudaMalloc(&pcuSrc, sizeof(float3) * nview))
		{
			throw ("pcuSrc allocation failed");
		}
	}
	catch (exception& e)
	{
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
		if (pcuSrc != NULL) cudaFree(pcuSrc);

		ostringstream oss;
		oss << "cDistanceDrivenTomoProjection() failed: " << e.what()
				<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw(oss.str().c_str());
	}

	cudaMemset(pcuImg, 0, sizeof(float) * nx * ny * nz * nBatches * nChannels);
	cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuSrc, src, sizeof(float3) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(pcuPrj, prj, sizeof(float) * nu * nview * nv * nBatches * nChannels, cudaMemcpyHostToDevice);

	DistanceDrivenTomo projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nview, nv, du, dv, off_u, off_v, 0, 0, 0);

	projector.BackprojectionTomo(pcuImg, pcuPrj, pcuDetCenter, pcuSrc);
	cudaMemcpy(img, pcuImg, sizeof(float) * nx * ny * nz * nBatches * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuPrj);
	cudaFree(pcuImg);
	cudaFree(pcuDetCenter);
	cudaFree(pcuSrc);

}


