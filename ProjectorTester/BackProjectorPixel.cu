#include "fan.h"

const static int zBatchBP = 5;
__global__ void BackprojectionPixelTextureKernel(float* pImg, cudaTextureObject_t texPrj,
		float* pDeg, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso)
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
		float sin_a = fabs(__sinf(a));
		if (sin_a > 1e-6f)
		{
			dist = fminf(dy / __cosf(a), dx / sin_a);
		}
		else
		{
			dist = dy / __cosf(a);
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
			pImg[ix * nz * ny + iy * nz + (iz+zStart)] += val[iz];
		}
	}

}

void fan3D::BackprojectionPixelTexture(float* pcuImg, const float* pcuPrj)
{
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), nz);

	cudaMemcpy(m_pcuPrj, pcuPrj, sizeof(float) * nu * nview * nv, cudaMemcpyDeviceToDevice);

	bindPrjTex(m_pcuPrj);
	BackprojectionPixelTextureKernel<<<blocks, threads>>>(pcuImg, m_texPrj, m_pcuDeg,
			nx, ny, nz, dx, dy, dz, nu, nview, nv, da, dv, off_a, off_v, dsd, dso);
	cudaDeviceSynchronize();

}

