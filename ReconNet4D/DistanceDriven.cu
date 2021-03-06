#include "DistanceDriven.h"
#include "cudaMath.h"
#include "Siddon.h"

#include <stdexcept>
#include <exception>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

// origin of image coordinate is the corner of the first pixel, the same with texture coordinate.
__device__ float2 PhysicsToImg(float2 pt, const Grid grid)
{
	pt.x = (pt.x - grid.cx) / grid.dx + grid.nx / 2.0f;
	pt.y = (pt.y - grid.cy) / grid.dy + grid.ny / 2.0f;

	return pt;
}

// the dst is realigned to make channel a higher dimension than x, y, z for later texture binding
__global__ void AccumulateKernelX(float* dst, const float* src, int nx, int ny, int nz, int nc)
{
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (iy >= ny || iz >= nz)
	{
		return;
	}

	dst += iy * nz + iz;
	src += iy * nz * nc + iz * nc;
	dst[0] = 0;
	for (int ix = 0; ix < nx; ix++)
	{
		dst[(ix+1) * ny * nz] = dst[ix * ny * nz] + src[ix * ny * nz * nc];
	}
}

// the dst is realigned to make channel a higher dimension than x, y, z for later texture binding
__global__ void AccumulateKernelY(float* dst, const float* src, int nx, int ny, int nz, int nc)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (ix >= nx || iz >= nz)
	{
		return;
	}

	dst += ix * (ny + 1) * nz + iz;
	src += ix * ny * nz * nc + iz * nc;
	dst[0] = 0;
	for (int iy = 0; iy < ny; iy++)
	{
		dst[(iy + 1) * nz] = dst[iy * nz] + src[iy * nz * nc];
	}
}

// iviews - the list of iview where abs(src.x) > abs(src.y)
// nValidViews - length of iviews
__global__ void DDFPFanKernelX(float* pPrjs, cudaTextureObject_t texAcc, const float* pDeg,
		const int* iviews, int nValidViews,
		int nview, int nc, const Grid grid, const Detector det,
		float dsd, float dso)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int ind = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || ind >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[ind];
	float deg = pDeg[iview];

	float cosDeg = __cosf(deg);
	float sinDeg = __sinf(deg);
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// calculate the coordinates of the detector cell's edges
	float2 src = make_float2(dso * sinDeg, -dso * cosDeg);
	float2 dst1 = make_float2(dsd * __sinf(a - det.du / 2), -dso + dsd * __cosf(a - det.du / 2));
	float2 dst2 = make_float2(dsd * __sinf(a + det.du / 2), -dso + dsd * __cosf(a + det.du / 2));
	dst1 = make_float2(dst1.x * cosDeg - dst1.y * sinDeg, dst1.x * sinDeg + dst1.y * cosDeg);
	dst2 = make_float2(dst2.x * cosDeg - dst2.y * sinDeg, dst2.x * sinDeg + dst2.y * cosDeg);

	// convert to image coordinate
	src = PhysicsToImg(src, grid);
	dst1 = PhysicsToImg(dst1, grid);
	dst2 = PhysicsToImg(dst2, grid);
	z = (z - grid.cz) / grid.dz + grid.nz / 2.f;



	// make sure dst1.x < dst2.x
	if (dst1.x > dst2.x)
	{
		float2 temp = dst1;
		dst1 = dst2;
		dst2 = temp;
	}

	float val = 0;
	float r1 = (dst1.x - src.x) / (dst1.y - src.y);
	float r2 = (dst2.x - src.x) / (dst2.y - src.y);
	// calculate the intersection with at each y
	for (int iy = 0; iy < grid.ny; iy++)
	{
		float x1 = src.x + r1 * (iy - src.y);
		float x2 = src.x + r2 * (iy - src.y);

		// the center of x0 of texAcc is the integral at the left border of x0 of image, since x1 and x2 are coordinates on the image, so
		// an offset of +0.5 should be added when fetching the integral value
		val += (tex3D<float>(texAcc, z, iy, x2 + 0.5f) - tex3D<float>(texAcc, z, iy, x1 + 0.5f)) / (x2 - x1);
	}

	// normalize by length
	val *= grid.dy / fabsf(__cosf(deg - a));

	pPrjs[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc] = val;

}

__global__ void DDFPFanKernelY(float* pPrjs, cudaTextureObject_t texAcc, const float* pDeg,
		const int* iviews, int nValidViews,
		int nview, int nc, const Grid grid, const Detector det,
		float dsd, float dso)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int ind = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || ind >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[ind];
	float deg = pDeg[iview];

	float cosDeg = __cosf(deg);
	float sinDeg = __sinf(deg);
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// calculate the coordinates of the detector cell's edges
	float2 src = make_float2(dso * sinDeg, -dso * cosDeg);
	float2 dst1 = make_float2(dsd * __sinf(a - det.du / 2), -dso + dsd * __cosf(a - det.du / 2));
	float2 dst2 = make_float2(dsd * __sinf(a + det.du / 2), -dso + dsd * __cosf(a + det.du / 2));
	dst1 = make_float2(dst1.x * cosDeg - dst1.y * sinDeg, dst1.x * sinDeg + dst1.y * cosDeg);
	dst2 = make_float2(dst2.x * cosDeg - dst2.y * sinDeg, dst2.x * sinDeg + dst2.y * cosDeg);

	// convert to image coordinate
	src = PhysicsToImg(src, grid);
	dst1 = PhysicsToImg(dst1, grid);
	dst2 = PhysicsToImg(dst2, grid);
	z = (z - grid.cz) / grid.dz + grid.nz / 2.f;

	// make sure dst1.y < dst2.y
	if (dst1.y > dst2.y)
	{
		float2 temp = dst1;
		dst1 = dst2;
		dst2 = temp;
	}

	float val = 0;
	float r1 = (dst1.y - src.y) / (dst1.x - src.x);
	float r2 = (dst2.y - src.y) / (dst2.x - src.x);
	// calculate the intersection with at each x
	for (int ix = 0; ix < grid.nx; ix++)
	{
		float y1 = src.y + r1 * (ix - src.x);
		float y2 = src.y + r2 * (ix - src.x);

		// the center of x0 of texAcc is the integral at the left border of x0 of image, since x1 and x2 are coordinates on the image, so
		// an offset of +0.5 should be added when fetching the integral value
		val += (tex3D<float>(texAcc, z, y2 + 0.5f, ix) - tex3D<float>(texAcc, z, y1 + 0.5f, ix)) / (y2 - y1);
	}

	// normalize by length
	val *= grid.dx / fabsf(__sinf(deg - a));

	pPrjs[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc] = val;

}

void DistanceDrivenFan::Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg)
{
	// allocate textures for accumulated images
	float* pAccX = NULL;
	float* pAccY = NULL;
	int* cuIviewsX = NULL;
	int* cuIviewsY = NULL;
	cudaArray_t arrAccX = NULL;
	cudaArray_t arrAccY = NULL;
	cudaTextureObject_t texAccX = 0;
	cudaTextureObject_t texAccY = 0;

	try
	{
		if (cudaSuccess != cudaMalloc(&pAccX, sizeof(float) * (nx+1) * ny * nz * nBatches * nChannels))
		{
			throw runtime_error("pAccX allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pAccY, sizeof(float) * nx * (ny + 1) * nz * nBatches * nChannels))
		{
			throw runtime_error("pAccY allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&cuIviewsX, sizeof(int) * nview))
		{
			throw runtime_error("cuIviewsX allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&cuIviewsY, sizeof(int) * nview))
		{
			throw runtime_error("cuIviewsY allocation failed");
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		if (cudaSuccess != cudaMalloc3DArray(&arrAccX, &channelDesc, make_cudaExtent(nz, ny, nx+1)))
		{
			throw runtime_error("arrAccX allocation failed");
		}

		if (cudaSuccess != cudaMalloc3DArray(&arrAccY, &channelDesc, make_cudaExtent(nz, ny+1, nx)))
		{
			throw runtime_error("arrAccY allocation failed");
		}

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = arrAccX;

		if (cudaSuccess != cudaCreateTextureObject(&texAccX, &resDesc, &texDesc, NULL))
		{
			throw std::runtime_error("texAccX binding failure!");
		}

		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeBorder;
		resDesc.res.array.array = arrAccY;

		if (cudaSuccess != cudaCreateTextureObject(&texAccY, &resDesc, &texDesc, NULL))
		{
			throw std::runtime_error("texAccY binding failure!");
		}

	}
	catch (exception &e)
	{
		if (pAccX != NULL) cudaFree(pAccX);
		if (pAccY != NULL) cudaFree(pAccY);
		if (cuIviewsX != NULL) cudaFree(cuIviewsX);
		if (cuIviewsY != NULL) cudaFree(cuIviewsY);
		if (arrAccX != NULL) cudaFreeArray(arrAccX);
		if (arrAccY != NULL) cudaFreeArray(arrAccY);
		if (texAccX != 0) cudaDestroyTextureObject(texAccX);
		if (texAccY != 0) cudaDestroyTextureObject(texAccY);

		ostringstream oss;
		oss << "DistanceDrivenFan::Projection Error: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

	// split the angles into set for accX and accY
	float* pDegs = new float [nview];
	int* iviewsX = new int [nview];
	int* iviewsY = new int [nview];
	int nValidViewsX = 0;
	int nValidViewsY = 0;

	cudaMemcpy(pDegs, pcuDeg, sizeof(float) * nview, cudaMemcpyDeviceToHost);
	for (int iview = 0; iview < nview; iview++)
	{
		float deg = pDegs[iview];
		if (abs(cosf(deg)) > abs(sinf(deg)))
		{
			iviewsX[nValidViewsX] = iview;
			nValidViewsX++;
		}
		else
		{
			iviewsY[nValidViewsY] = iview;
			nValidViewsY++;
		}
	}
	cudaMemcpy(cuIviewsX, iviewsX, sizeof(int) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(cuIviewsY, iviewsY, sizeof(int) * nview, cudaMemcpyHostToDevice);

	delete [] iviewsX;
	delete [] iviewsY;
	delete [] pDegs;

	Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
	Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

	// step 1: calculate accumulated images
	dim3 threadX(1,256,1);
	dim3 blockX(1, ceilf(ny / 256.0f), nz);
	dim3 threadY(256,1,1);
	dim3 blockY(ceilf(nx / 256.0f), 1, nz);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			// pAccX and pAccY has the dimension in order (batch, channel, x, y, z) so that can be copied to textures
			AccumulateKernelX<<<blockX, threadX>>>(pAccX + (ib * nChannels + ic) * (nx + 1) * ny * nz,
					pcuImg + ib * nx * ny * nz * nChannels + ic , nx, ny, nz, nChannels);
			AccumulateKernelY<<<blockY, threadY>>>(pAccY + (ib * nChannels + ic) * nx * (ny + 1) * nz,
					pcuImg + ib * nx * ny * nz * nChannels + ic , nx, ny, nz, nChannels);
		}
	}
	cudaDeviceSynchronize();

	// step 2: interpolation
//	dim3 thread(256, 1, 1);
//	dim3 block(ceilf(nu / 256.f), 1, 1);
	dim3 threadsX, blocksX, threadsY, blocksY;
	GetThreadsForXY(threadsX, blocksX, nu, nValidViewsX, nv);
	GetThreadsForXY(threadsY, blocksY, nu, nValidViewsY, nv);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			// copy to array
			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr = make_cudaPitchedPtr(pAccX + (ib * nChannels + ic) * (nx + 1) * ny * nz, nz * sizeof(float), nz, ny);
			copyParams.dstArray = arrAccX;
			copyParams.extent = make_cudaExtent(nz, ny, nx + 1);
			copyParams.kind = cudaMemcpyDeviceToDevice;
			cudaMemcpy3DAsync(&copyParams, m_stream);

			copyParams.srcPtr = make_cudaPitchedPtr(pAccY + (ib * nChannels + ic) * nx * (ny + 1) * nz, nz * sizeof(float), nz, ny + 1);
			copyParams.dstArray = arrAccY;
			copyParams.extent = make_cudaExtent(nz, ny + 1, nx);
			copyParams.kind = cudaMemcpyDeviceToDevice;
			cudaMemcpy3DAsync(&copyParams, m_stream);

			DDFPFanKernelX<<<blocksX, threadsX>>>(pcuPrj + ib * nu * nview * nv * nChannels + ic,
					texAccX, pcuDeg, cuIviewsX, nValidViewsX, nview, nChannels, grid, det, dsd, dso);

			DDFPFanKernelY<<<blocksY, threadsY>>>(pcuPrj + ib * nu * nview * nv * nChannels + ic,
					texAccY, pcuDeg, cuIviewsY, nValidViewsY, nview, nChannels, grid, det, dsd, dso);

			cudaDeviceSynchronize();
		}
	}


	if (pAccX != NULL) cudaFree(pAccX);
	if (pAccY != NULL) cudaFree(pAccY);
	if (cuIviewsX != NULL) cudaFree(cuIviewsX);
	if (cuIviewsY != NULL) cudaFree(cuIviewsY);
	if (arrAccX != NULL) cudaFreeArray(arrAccX);
	if (arrAccY != NULL) cudaFreeArray(arrAccY);
	if (texAccX != 0) cudaDestroyTextureObject(texAccX);
	if (texAccY != 0) cudaDestroyTextureObject(texAccY);
}

extern "C" void cDistanceDrivenFanProjection(float* prj, const float* img, const float* deg,
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
		oss << "cDistanceDrivenFanProjection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	DistanceDrivenFan projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeProjector);

	projector.Projection(pcuImg, pcuPrj, pcuDeg);
	cudaMemcpy(prj, pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuDeg);

}

__global__ void PreweightBPKernelX(float* pPrjs, const float* pDeg, int* iviews, int nValidViews,
		int nview, int nc, const Detector det, float dy)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iviewInd = blockIdx.y * blockDim.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iviewInd >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[iviewInd];
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float deg = pDeg[iview];

	pPrjs[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc] *= dy / fabsf(__cosf(deg - a));
}

__global__ void PreweightBPKernelY(float* pPrjs, const float* pDeg, int* iviews, int nValidViews,
		int nview, int nc, const Detector det, float dx)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iviewInd = blockIdx.y * blockDim.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iviewInd >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[iviewInd];
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float deg = pDeg[iview];

	pPrjs[iu * nview * det.nv * nc + iview * det.nv * nc + iv * nc] *= dx / fabsf(__sinf(deg - a));
}

__device__ float GetProjectionOnDetector(float x, float y, float dsd, float dso, float cosDeg, float sinDeg)
{
	float rx =  x * cosDeg + y * sinDeg;
	float ry = -x * sinDeg + y * cosDeg;
	return atanf(rx / (ry + dso));

}

__device__ float GetFBPWeight(float x, float y, float dsd, float dso, float cosDeg, float sinDeg)
{
	float rx =  x * cosDeg + y * sinDeg;
	float ry = -x * sinDeg + y * cosDeg;

	return dso*dso / (rx*rx + (dso+ry)*(dso+ry));

}

// iviews - the list of iview where abs(src.x) > abs(src.y)
// nValidViews - length of iviews
// isFBP - if true: weight the views by length factor then BP
//			if false: direct BP then weight according to FBP
__global__ void DDBPFanKernelX(float* pImg, cudaTextureObject_t texAcc, const float* pDeg,
		const int* iviews, int nValidViews,
		int nview, int nc, const Grid grid, const Detector det,
		float dsd, float dso, bool isFBP)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= grid.nx || iy >= grid.ny || iz >= grid.nz)
	{
		return;
	}

	float x = (ix - (grid.nx - 1) / 2.0f) * grid.dx;
	float x1 = x - grid.dx / 2.0f;
	float x2 = x + grid.dx / 2.0f;
	float y = (iy - (grid.ny - 1) / 2.0f) * grid.dy;
	float iv = (iz - (grid.nz - 1) / 2.0f) * grid.dz / det.dv + det.off_v + (det.nv - 1.0f) / 2.f;

	for (int ind = 0; ind < nValidViews; ind++)
	{
		int iview = iviews[ind];
		float deg = pDeg[iview];

		float cosDeg = __cosf(deg);
		float sinDeg = __sinf(deg);

		// For the u direction, the origin is the border of pixel 0
		// It is equivalent to the center of pixel 0 for the accumulated image, so there is 0.5 offset on the texture.
		float a1 = GetProjectionOnDetector(x1, y, dsd, dso, cosDeg, sinDeg);
		a1 = -(a1 / det.du + det.off_u) + det.nu / 2.0f;
		float a2 = GetProjectionOnDetector(x2, y, dsd, dso, cosDeg, sinDeg);
		a2 = -(a2 / det.du + det.off_u) + det.nu / 2.0f;

		// make sure a1 < a2
		if (a1 > a2)
		{
			float t = a1;
			a1 = a2;
			a2 = t;
		}

		float val = (tex3D<float>(texAcc, iv + 0.5f, iview + 0.5f, a2 + 0.5f) - tex3D<float>(texAcc, iv + 0.5f, iview + 0.5f, a1 + 0.5f)) / (a2 - a1);

		if (isFBP)
		{
			pImg[ix * grid.ny * grid.nz * nc + iy * grid.nz * nc + iz * nc] += val * GetFBPWeight(x, y, dsd, dso, cosDeg, sinDeg);
		}
		else
		{
			pImg[ix * grid.ny * grid.nz * nc + iy * grid.nz * nc + iz * nc] += val;
		}

	}

}

__global__ void DDBPFanKernelY(float* pImg, cudaTextureObject_t texAcc, const float* pDeg,
		const int* iviews, int nValidViews,
		int nview, int nc, const Grid grid, const Detector det,
		float dsd, float dso, bool isFBP)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= grid.nx || iy >= grid.ny || iz >= grid.nz)
	{
		return;
	}

	float x = (ix - (grid.nx - 1) / 2.0f) * grid.dx;
	float y = (iy - (grid.ny - 1) / 2.0f) * grid.dy;
	float y1 = y - grid.dy / 2.0f;
	float y2 = y + grid.dy / 2.0f;
	float iv = (iz - (grid.nz - 1) / 2.0f) * grid.dz / det.dv + det.off_v + (det.nv - 1.0f) / 2.f;

	for (int ind = 0; ind < nValidViews; ind++)
	{
		int iview = iviews[ind];
		float deg = pDeg[iview];

		float cosDeg = __cosf(deg);
		float sinDeg = __sinf(deg);

		// For the u direction, the origin is the border of pixel 0
		// It is equivalent to the center of pixel 0 for the accumulated image, so there is 0.5 offset on the texture.
		float a1 = GetProjectionOnDetector(x, y1, dsd, dso, cosDeg, sinDeg);
		a1 = -(a1 / det.du + det.off_u) + det.nu / 2.0f;
		float a2 = GetProjectionOnDetector(x, y2, dsd, dso, cosDeg, sinDeg);
		a2 = -(a2 / det.du + det.off_u) + det.nu / 2.0f;

		// make sure a1 < a2
		if (a1 > a2)
		{
			float t = a1;
			a1 = a2;
			a2 = t;
		}

		float val = (tex3D<float>(texAcc, iv + 0.5f, iview + 0.5f, a2 + 0.5f) - tex3D<float>(texAcc, iv + 0.5f, iview + 0.5f, a1 + 0.5f)) / (a2 - a1);

		if (isFBP)
		{
			pImg[ix * grid.ny * grid.nz * nc + iy * grid.nz * nc + iz * nc] += val * GetFBPWeight(x, y, dsd, dso, cosDeg, sinDeg);
		}
		else
		{
			pImg[ix * grid.ny * grid.nz * nc + iy * grid.nz * nc + iz * nc] += val;
		}

	}

}



// backprojection
void DistanceDrivenFan::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	float* pWeightedPrjs = NULL;
	float* pAccU = NULL;
	cudaArray_t arrAccU = NULL;
	cudaTextureObject_t texAccU = 0;
	int* cuIviewsX = NULL;
	int* cuIviewsY = NULL;
	bool isFBP = (typeProjector == 1);

	try
	{
		if (cudaSuccess != cudaMalloc(&pWeightedPrjs, sizeof(float) * nBatches * nChannels * nu * nview * nv))
		{
			throw runtime_error("pWeightedPrjs allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pAccU, sizeof(float) * nBatches * nChannels * (nu + 1) * nview * nv))
		{
			throw runtime_error("pAccU allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&cuIviewsX, sizeof(int) * nview))
		{
			throw runtime_error("cuIviewsX allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&cuIviewsY, sizeof(int) * nview))
		{
			throw runtime_error("cuIviewsY allocation failed");
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		if (cudaSuccess != cudaMalloc3DArray(&arrAccU, &channelDesc, make_cudaExtent(nv, nview, nu + 1)))
		{
			throw runtime_error("arrAccU allocation failed");
		}

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = arrAccU;

		if (cudaSuccess != cudaCreateTextureObject(&texAccU, &resDesc, &texDesc, NULL))
		{
			throw std::runtime_error("texAccU binding failure!");
		}

		cudaMemcpy(pWeightedPrjs, pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyDeviceToDevice);
	}
	catch (exception &e)
	{
		if (pAccU != NULL) cudaFree(pAccU);
		if (pWeightedPrjs != NULL) cudaFree(pWeightedPrjs);
		if (arrAccU != NULL) cudaFreeArray(arrAccU);
		if (texAccU != 0) cudaDestroyTextureObject(texAccU);
		if (cuIviewsX != NULL) cudaFree(cuIviewsX);
		if (cuIviewsY != NULL) cudaFree(cuIviewsY);

		ostringstream oss;
		oss << "DistanceDrivenFan::Backprojection error: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

	// separate x and y directions
	float* pDegs = new float [nview];
	int* iviewsX = new int [nview];
	int* iviewsY = new int [nview];
	int nValidViewsX = 0;
	int nValidViewsY = 0;

	cudaMemcpy(pDegs, pcuDeg, sizeof(float) * nview, cudaMemcpyDeviceToHost);
	for (int iview = 0; iview < nview; iview++)
	{
		float deg = pDegs[iview];
		if (abs(cosf(deg)) > abs(sinf(deg)))
		{
			iviewsX[nValidViewsX] = iview;
			nValidViewsX++;
		}
		else
		{
			iviewsY[nValidViewsY] = iview;
			nValidViewsY++;
		}
	}
	cudaMemcpy(cuIviewsX, iviewsX, sizeof(int) * nview, cudaMemcpyHostToDevice);
	cudaMemcpy(cuIviewsY, iviewsY, sizeof(int) * nview, cudaMemcpyHostToDevice);

	delete [] iviewsX;
	delete [] iviewsY;
	delete [] pDegs;

	Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
	Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

	// pre-weight for iterative BP to make it conjugate to FP
	dim3 blocksX, threadsX, blocksY, threadsY;
	if (!isFBP) // not FBP
	{
		GetThreadsForXY(threadsX, blocksX, nu, nValidViewsX, nv);
		GetThreadsForXY(threadsY, blocksY, nu, nValidViewsY, nv);
		for (int ib = 0; ib < nBatches; ib++)
		{
			for (int ic = 0; ic < nChannels; ic++)
			{
				PreweightBPKernelX<<<blocksX, threadsX>>>(pWeightedPrjs + ib * nu * nview * nv * nChannels + ic,
						pcuDeg, cuIviewsX, nValidViewsX, nview, nChannels, det, grid.dy);
				PreweightBPKernelY<<<blocksY, threadsY>>>(pWeightedPrjs + ib * nu * nview * nv * nChannels + ic,
						pcuDeg, cuIviewsY, nValidViewsY, nview, nChannels, det, grid.dx);
			}
		}
		cudaDeviceSynchronize();
	}

	// step 1: accumulate projections along u axis
	dim3 threadU(1,256,1);
	dim3 blockU(1, ceilf(nview / 256.0f), nv);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			AccumulateKernelX<<<blockU, threadU>>>(pAccU + (ib * nChannels + ic) * (nu + 1) * nview * nv,
					pWeightedPrjs + ib * nu * nview * nv * nChannels + ic , nu, nview, nv, nChannels);
		}
	}
	cudaDeviceSynchronize();

	// step 2: backprojection
	dim3 threads, blocks;
	GetThreadsForXY(threads, blocks, nx, ny, nz);
	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int ic = 0; ic < nChannels; ic++)
		{
			// copy to array
			cudaMemcpy3DParms copyParams = {0};
			copyParams.srcPtr = make_cudaPitchedPtr(pAccU + (ib * nChannels + ic) * (nu + 1) * nview * nv, nv * sizeof(float), nv, nview);
			copyParams.dstArray = arrAccU;
			copyParams.extent = make_cudaExtent(nv, nview, nu + 1);
			copyParams.kind = cudaMemcpyDeviceToDevice;
			cudaMemcpy3DAsync(&copyParams, m_stream);

			DDBPFanKernelX<<<blocks, threads>>>(pcuImg + ib * nx * ny * nz * nChannels + ic,
					texAccU, pcuDeg, cuIviewsX, nValidViewsX, nview, nChannels, grid, det, dsd, dso, isFBP);

			DDBPFanKernelY<<<blocks, threads>>>(pcuImg + ib * nx * ny * nz * nChannels + ic,
					texAccU, pcuDeg, cuIviewsY, nValidViewsY, nview, nChannels, grid, det, dsd, dso, isFBP);

			cudaDeviceSynchronize();
		}
	}


	if (pAccU != NULL) cudaFree(pAccU);
	if (pWeightedPrjs != NULL) cudaFree(pWeightedPrjs);
	if (arrAccU != NULL) cudaFreeArray(arrAccU);
	if (texAccU != 0) cudaDestroyTextureObject(texAccU);
	if (cuIviewsX != NULL) cudaFree(cuIviewsX);
	if (cuIviewsY != NULL) cudaFree(cuIviewsY);

}

extern "C" void cDistanceDrivenFanBackprojection(float* img, const float* prj, const float* deg,
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
		oss << "cDistanceDrivenFanBackprojection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	DistanceDrivenFan projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeProjector);

	projector.Backprojection(pcuImg, pcuPrj, pcuDeg);
	cudaMemcpy(img, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuDeg);

}
