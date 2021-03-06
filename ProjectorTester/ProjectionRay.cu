#include "fan.h"

static __device__ float2 PhysicsToImg(float2 pt, int nx, int ny, float dx, float dy)
{
	pt.x = pt.x / dx + nx / 2.0f;
	pt.y = pt.y / dy + ny / 2.0f;

	return pt;
}

__device__ float SoftTex3DFromTexture(cudaTextureObject_t texImg, float x, float y, float z, int nx, int ny, int nz)
{
	float3 w0;
	float3 w1;
	x -= 0.5f;
	y -= 0.5f;
	z -= 0.5f;
	int ix = (int)x;
	int iy = (int)y;
	int iz = (int)z;

	if (ix < 0 || ix >= nx)
	{
		w0.x = 0;
	}
	else
	{
		w0.x = ix + 1 - x;
	}
	if (ix + 1 < 0 || ix + 1 >= nx)
	{
		w1.x = 0;
	}
	else
	{
		w1.x = x - ix;
	}
	if (iy < 0 || iy >= ny)
	{
		w0.y = 0;
	}
	else
	{
		w0.y = iy + 1 - y;
	}
	if (iy + 1 < 0 || iy + 1 >= ny)
	{
		w1.y = 0;
	}
	else
	{
		w1.y = y - iy;
	}
	if (iz < 0 || iz >= nz)
	{
		w0.z = 0;
	}
	else
	{
		w0.z = iz + 1 - z;
	}
	if (iz + 1 < 0 || iz + 1 >= nz)
	{
		w1.z = 0;
	}
	else
	{
		w1.z = z - iz;
	}

	float val = 0;
	float w;
	if ((w = w0.x * w0.y * w0.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 0.5f, iy + 0.5f, ix + 0.5f) * w;
	}
	if ((w = w0.x * w0.y * w1.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 1.5f, iy + 0.5f, ix + 0.5f) * w;
	}
	if ((w = w0.x * w1.y * w0.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 0.5f, iy + 1.5f, ix + 0.5f) * w;
	}
	if ((w = w0.x * w1.y * w1.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 1.5f, iy + 1.5f, ix + 0.5f) * w;
	}
	if ((w = w1.x * w0.y * w0.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 0.5f, iy + 0.5f, ix + 1.5f) * w;
	}
	if ((w = w1.x * w0.y * w1.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 1.5f, iy + 0.5f, ix + 1.5f) * w;
	}
	if ((w = w1.x * w1.y * w0.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 0.5f, iy + 1.5f, ix + 1.5f) * w;
	}
	if ((w = w1.x * w1.y * w1.z) > 0)
	{
		val += tex3D<float>(texImg, iz + 1.5f, iy + 1.5f, ix + 1.5f) * w;
	}

	return val;

}

__global__ void ProjectionRasterTextureKernel(float* pPrj, cudaTextureObject_t texImg,
		float* pDeg, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= nu || iview >= nview)
	{
		return;
	}

	float cosDeg = __cosf(pDeg[iview]);
	float sinDeg = __sinf(pDeg[iview]);
	float a = (-(iu - (nu-1) / 2.0f) - off_a) * da;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);

	// src, dst points, and convert to image coordinate
	float2 src = make_float2(dso * sinDeg, -dso * cosDeg);
	float2 dstRel = make_float2(dsd * sin_a, -dso + dsd * cos_a);
	float2 dst = make_float2(dstRel.x * cosDeg - dstRel.y * sinDeg,
			dstRel.x * sinDeg + dstRel.y * cosDeg);
	src = PhysicsToImg(src, nx, ny, dx, dy);
	dst = PhysicsToImg(dst, nx, ny, dx, dy);
	float2 dDstSrc = make_float2(dst.x - src.x, dst.y - src.y);

	// get the intersection points
	float2 alpha1 = {0, 0};
	float2 alpha2 = {1, 1};
	if (fabsf(dDstSrc.x) > 1e-6)
	{
		alpha1.x = -src.x / dDstSrc.x;
		alpha2.x = (nx - src.x) / dDstSrc.x;
	}
	if (fabsf(dDstSrc.y) > 1e-6)
	{
		alpha1.y = -src.y / dDstSrc.y;
		alpha2.y = (ny - src.y) / dDstSrc.y;
	}

	float alphaMin = fmaxf(fminf(alpha1.x, alpha2.x), fminf(alpha1.y, alpha2.y));
	float alphaMax = fminf(fmaxf(alpha1.x, alpha2.x), fmaxf(alpha1.y, alpha2.y));
	if (alphaMax <= alphaMin)
	{
		return;
	}

	float dt = fminf(dx, dy);
	float dist = sqrtf(dDstSrc.x * dDstSrc.x + dDstSrc.y * dDstSrc.y);
	float dAlpha = dt / dist;
	float2 pt0 = make_float2(src.x + alphaMin * dDstSrc.x, src.y + alphaMin * dDstSrc.y);
	float2 dt2 = make_float2(dAlpha * dDstSrc.x, dAlpha * dDstSrc.y);

	register float val = 0;
	register float rz = (iv - (nv-1) / 2.0f + off_v) * dv / dz + (nz - 1) / 2.0f;
	for (float alpha = alphaMin; alpha <= alphaMax; alpha += dAlpha)
	{
		val += tex3D<float>(texImg, rz + 0.5f, pt0.y, pt0.x) * dt;
//		val += SoftTex3DFromTexture(texImg, pt0.x, pt0.y, rz + 0.5f, nx, ny, nz) * dt;

		pt0.x += dt2.x;
		pt0.y += dt2.y;
	}

	pPrj[iu * nv * nview + iview * nv + iv] += val;
}

void fan3D::ProjectionRasterTexture(float* pcuPrj, const float* pcuImg)
{
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), nv);

	cudaMemcpy(m_pcuImg, pcuImg, sizeof(float) * nx * ny * nz, cudaMemcpyDeviceToDevice);

	bindImgTex(m_pcuImg);
	ProjectionRasterTextureKernel<<<blocks, threads>>>(pcuPrj, m_texImg, m_pcuDeg,
			nx, ny, nz, dx, dy, dz, nu, nview, nv, da, dv, off_a, off_v, dsd, dso);
	cudaDeviceSynchronize();

}

__device__ float SoftTex3D(const float* img, float x, float y, float z, int nx, int ny, int nz)
{
	float3 w0;
	float3 w1;
	x -= 0.5f;
	y -= 0.5f;
	z -= 0.5f;
	int ix = (int)x;
	int iy = (int)y;
	int iz = (int)z;

	if (ix < 0 || ix >= nx)
	{
		w0.x = 0;
	}
	else
	{
		w0.x = ix + 1 - x;
	}
	if (ix + 1 < 0 || ix + 1 >= nx)
	{
		w1.x = 0;
	}
	else
	{
		w1.x = x - ix;
	}
	if (iy < 0 || iy >= ny)
	{
		w0.y = 0;
	}
	else
	{
		w0.y = iy + 1 - y;
	}
	if (iy + 1 < 0 || iy + 1 >= ny)
	{
		w1.y = 0;
	}
	else
	{
		w1.y = y - iy;
	}
	if (iz < 0 || iz >= nz)
	{
		w0.z = 0;
	}
	else
	{
		w0.z = iz + 1 - z;
	}
	if (iz + 1 < 0 || iz + 1 >= nz)
	{
		w1.z = 0;
	}
	else
	{
		w1.z = z - iz;
	}

	int ind = ix * ny * nz + iy * nz + iz;
	int idz = 1;
	int idy = nz;
	int idx = ny * nz;

	float val = 0;
	float w;
	if ((w = w0.x * w0.y * w0.z) > 0)
	{
		val += img[ind] * w;
	}
	if ((w = w0.x * w0.y * w1.z) > 0)
	{
		val += img[ind + idz] * w;
	}
	if ((w = w0.x * w1.y * w0.z) > 0)
	{
		val += img[ind + idy] * w;
	}
	if ((w = w0.x * w1.y * w1.z) > 0)
	{
		val += img[ind + idy + idz] * w;
	}
	if ((w = w1.x * w0.y * w0.z) > 0)
	{
		val += img[ind + idx] * w;
	}
	if ((w = w1.x * w0.y * w1.z) > 0)
	{
		val += img[ind + idx + idz] * w;
	}
	if ((w = w1.x * w1.y * w0.z) > 0)
	{
		val += img[ind + idx + idy] * w;
	}
	if ((w = w1.x * w1.y * w1.z) > 0)
	{
		val += img[ind + idx + idy + idz] * w;
	}

	return val;

}


__device__ float SoftTex3DNoCheck(const float* img, float x, float y, float z, int nx, int ny, int nz)
{
	float3 w0;
	float3 w1;
	x -= 0.5f;
	y -= 0.5f;
	z -= 0.5f;
	int ix = (int)x;
	int iy = (int)y;
	int iz = (int)z;

	w0.x = ix + 1 - x;
	w1.x = x - ix;
	w0.y = iy + 1 - y;
	w1.y = y - iy;
	w0.z = iz + 1 - z;
	w1.z = z - iz;

	int ind = ix * ny * nz + iy * nz + iz;
	int idz = 1;
	int idy = nz;
	int idx = ny * nz;

	if (ind < 0)
	{
		ind = 0;
	}
	else if (ind >= nx * ny * nz - idx - idy - idz)
	{
		ind = nx * ny * nz - idx - idy - idz - 1;
	}

	float val = img[ind] * w0.x * w0.y * w0.z
			+ img[ind + idz] * w0.x * w0.y * w1.z
			+ img[ind + idy] * w0.x * w1.y * w0.z
			+ img[ind + idy + idz] * w0.x * w1.y * w1.z
			+ img[ind + idx] * w1.x * w0.y * w0.z
			+ img[ind + idx + idz] * w1.x * w0.y * w1.z
			+ img[ind + idx + idy] * w1.x * w1.y * w0.z
			+ img[ind + idx + idy + idz] * w1.x * w1.y * w1.z;

	return val;

}

__global__ void ProjectionRasterKernel(float* pPrj, const float* pcuImg,
		float* pDeg, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.y * blockIdx.y + threadIdx.y;
	int iv = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= nu || iview >= nview)
	{
		return;
	}

	float cosDeg = __cosf(pDeg[iview]);
	float sinDeg = __sinf(pDeg[iview]);
	float a = (-(iu - (nu-1) / 2.0f) - off_a) * da;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);

	// src, dst points, and convert to image coordinate
	float2 src = make_float2(dso * sinDeg, -dso * cosDeg);
	float2 dstRel = make_float2(dsd * sin_a, -dso + dsd * cos_a);
	float2 dst = make_float2(dstRel.x * cosDeg - dstRel.y * sinDeg,
			dstRel.x * sinDeg + dstRel.y * cosDeg);
	src = PhysicsToImg(src, nx, ny, dx, dy);
	dst = PhysicsToImg(dst, nx, ny, dx, dy);
	float2 dDstSrc = make_float2(dst.x - src.x, dst.y - src.y);

	// get the intersection points
	float2 alpha1 = {0, 0};
	float2 alpha2 = {1, 1};
	if (fabsf(dDstSrc.x) > 1e-6)
	{
		alpha1.x = -src.x / dDstSrc.x;
		alpha2.x = (nx - src.x) / dDstSrc.x;
	}
	if (fabsf(dDstSrc.y) > 1e-6)
	{
		alpha1.y = -src.y / dDstSrc.y;
		alpha2.y = (ny - src.y) / dDstSrc.y;
	}

	float alphaMin = fmaxf(fminf(alpha1.x, alpha2.x), fminf(alpha1.y, alpha2.y));
	float alphaMax = fminf(fmaxf(alpha1.x, alpha2.x), fmaxf(alpha1.y, alpha2.y));
	if (alphaMax <= alphaMin)
	{
		return;
	}

	float dt = fminf(dx, dy);
	float dist = sqrtf(dDstSrc.x * dDstSrc.x + dDstSrc.y * dDstSrc.y);
	float dAlpha = dt / dist;
	float2 pt0 = make_float2(src.x + alphaMin * dDstSrc.x, src.y + alphaMin * dDstSrc.y);
	float2 dt2 = make_float2(dAlpha * dDstSrc.x, dAlpha * dDstSrc.y);

	register float val = 0;
	register float rz = (iv - (nv-1) / 2.0f + off_v) * dv / dz + (nz - 1) / 2.0f;
	for (float alpha = alphaMin; alpha <= alphaMax; alpha += dAlpha)
	{
		val += SoftTex3D(pcuImg, pt0.x, pt0.y, rz + 0.5f, nx, ny, nz) * dt;

		pt0.x += dt2.x;
		pt0.y += dt2.y;
	}

	pPrj[iu * nv * nview + iview * nv + iv] += val;
}

void fan3D::ProjectionRaster(float* pcuPrj, const float* pcuImg)
{
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), nv);

	ProjectionRasterKernel<<<blocks, threads>>>(pcuPrj, pcuImg, m_pcuDeg,
			nx, ny, nz, dx, dy, dz, nu, nview, nv, da, dv, off_a, off_v, dsd, dso);
	cudaDeviceSynchronize();

}

