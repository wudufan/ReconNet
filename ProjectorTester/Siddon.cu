#include "fan.h"

#define eps 1e-6f

// struct for siddon tracing
struct SiddonTracingVars
{
	float3 alpha;			// the position of next intersection
	float3 dAlpha;			// the step size between intersections
	int3 ng;				// the grid position
	int3 isPositive;		// if the tracing direction is positive (=1) or not (=0)
	float alphaNow;			// the current tracing position
	float alphaPrev;		// the previous tracing position
	float rayLength;		// total length of the ray
};

static __device__ float3 PhysicsToImg(float3 pt, int nx, int ny, int nz, float dx, float dy, float dz)
{
	pt.x = pt.x / dx + nx / 2.0f;
	pt.y = pt.y / dy + ny / 2.0f;
	pt.z = pt.z / dz + nz / 2.0f;

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
static __device__ int InboundFirstVoxel(float pt0v, float dDstSrcv, int gridnv)
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
static __device__ float OutboundFirstVoxel(int ngv, float dDstSrcv)
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
static __device__ void GetAlpha(float& alphav, float& dAlphav, float pt1v, float srcv, float dDstSrcv)
{
	if (fabsf(dDstSrcv) > eps)
	{
		alphav = (pt1v - srcv) / dDstSrcv;
		dAlphav = 1 / fabsf(dDstSrcv);
	}
}

__device__ SiddonTracingVars InitializeSiddon(float3 src, float3 dst,
		int nx, int ny, int nz, float dx, float dy, float dz)
{
	SiddonTracingVars var;

	float dist = sqrtf((src.x - dst.x) * (src.x - dst.x) +
			(src.y - dst.y) * (src.y - dst.y) +
			(src.z - dst.z) * (src.z - dst.z));

	src = PhysicsToImg(src, nx, ny, nz, dx, dy, dz);
	dst = PhysicsToImg(dst, nx, ny, nz, dx, dy, dz);

	float3 dDstSrc = make_float3(dst.x - src.x, dst.y - src.y, dst.z - src.z);

	// intersection between ray and grid
	float3 a0 = make_float3(1.0f, 1.0f, 1.0f);
	float3 a1 = make_float3(0.0f, 0.0f, 0.0f);
	if (!InboundAlpha(a0.x, a1.x, dDstSrc.x, src.x, nx) ||
		!InboundAlpha(a0.y, a1.y, dDstSrc.y, src.y, ny) ||
		!InboundAlpha(a0.z, a1.z, dDstSrc.z, src.z, nz))
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
	int3 ng0 = make_int3(InboundFirstVoxel(pt0.x, dDstSrc.x, nx),
		InboundFirstVoxel(pt0.y, dDstSrc.y, ny),
		InboundFirstVoxel(pt0.z, dDstSrc.z, nz));

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

__device__ float SiddonRayTracing(float* pPrj, const float* pImg, float3 src, float3 dst,
		int nx, int ny, int nz, float dx, float dy, float dz)
{
	SiddonTracingVars var = InitializeSiddon(src, dst, nx, ny, nz, dx, dy, dz);

	if (var.ng.x < 0 || var.ng.x >= nx || var.ng.y < 0 || var.ng.y >= ny || var.ng.z < 0 || var.ng.z >= nz)
	{
		// no intersections
		return 0;
	}

	int move = 0;
	bool isTracing = true;
	float val = 0;
	int nyz = ny * nz;
	pImg += var.ng.x * nyz + var.ng.y * nz + var.ng.z;

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
				move = nyz;
				if (var.ng.x >= nx) isTracing = false;
			}
			else
			{
				var.ng.x--;
				move = -nyz;
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
				move = nz;
				if (var.ng.y >= ny) isTracing = false;
			}
			else
			{
				var.ng.y--;
				move = -nz;
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
				move = 1;
				if (var.ng.z >= nz) isTracing = false;
			}
			else
			{
				var.ng.z--;
				move = -1;
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

__global__ void ProjectionSiddonKernel(float* pPrj, const float* pImg,
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
	float z = (iv - (nv-1) / 2.0f + off_v) * dv;

	// src, dst points, and convert to image coordinate
	float3 src = make_float3(dso * sinDeg, -dso * cosDeg, z);
	float3 dstRel = make_float3(dsd * sin_a, -dso + dsd * cos_a, z);
	float3 dst = make_float3(dstRel.x * cosDeg - dstRel.y * sinDeg, dstRel.x * sinDeg + dstRel.y * cosDeg, z);

	SiddonRayTracing(pPrj + iu * nview * nv + iview * nv + iv, pImg, src, dst, nx, ny, nz, dx, dy, dz);

}

void fan3D::ProjectionSiddon(float* pcuPrj, const float* pcuImg)
{
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), nv);

	ProjectionSiddonKernel<<<blocks, threads>>>(pcuPrj, pcuImg, m_pcuDeg,
			nx, ny, nz, dx, dy, dz, nu, nview, nv, da, dv, off_a, off_v, dsd, dso);
	cudaDeviceSynchronize();

}

__device__ float SiddonRayTracingTexture(float* pPrj, cudaTextureObject_t texImg, float3 src, float3 dst,
		int nx, int ny, int nz, float dx, float dy, float dz)
{
	SiddonTracingVars var = InitializeSiddon(src, dst, nx, ny, nz, dx, dy, dz);

	if (var.ng.x < 0 || var.ng.x >= nx || var.ng.y < 0 || var.ng.y >= ny || var.ng.z < 0 || var.ng.z >= nz)
	{
		// no intersections
		return 0;
	}

	bool isTracing = true;
	float val = 0;
	int3 currentNg = var.ng;

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
				if (var.ng.x >= nx) isTracing = false;
			}
			else
			{
				var.ng.x--;
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
				if (var.ng.y >= ny) isTracing = false;
			}
			else
			{
				var.ng.y--;
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
				if (var.ng.z >= nz) isTracing = false;
			}
			else
			{
				var.ng.z--;
				if (var.ng.z < 0) isTracing = false;
			}
		}

		val += tex3D<float>(texImg, currentNg.z + 0.5f, currentNg.y + 0.5f, currentNg.x + 0.5f)
				* (var.alphaNow - var.alphaPrev);
		var.alphaPrev = var.alphaNow;
		currentNg = var.ng;
	}

	*pPrj += val;

	return var.rayLength;
}


__global__ void ProjectionSiddonTextureKernel(float* pPrj, cudaTextureObject_t texImg,
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
	float z = (iv - (nv-1) / 2.0f + off_v) * dv;

	// src, dst points, and convert to image coordinate
	float3 src = make_float3(dso * sinDeg, -dso * cosDeg, z);
	float3 dstRel = make_float3(dsd * sin_a, -dso + dsd * cos_a, z);
	float3 dst = make_float3(dstRel.x * cosDeg - dstRel.y * sinDeg, dstRel.x * sinDeg + dstRel.y * cosDeg, z);

	SiddonRayTracingTexture(pPrj + iu * nview * nv + iview * nv + iv, texImg,
			src, dst, nx, ny, nz, dx, dy, dz);

}

void fan3D::ProjectionSiddonTexture(float* pcuPrj, const float* pcuImg)
{
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), nv);

	cudaMemcpy(m_pcuImg, pcuImg, sizeof(float) * nx * ny * nz, cudaMemcpyDeviceToDevice);

	bindImgTex(m_pcuImg);
	ProjectionSiddonTextureKernel<<<blocks, threads>>>(pcuPrj, m_texImg, m_pcuDeg,
			nx, ny, nz, dx, dy, dz, nu, nview, nv, da, dv, off_a, off_v, dsd, dso);
	cudaDeviceSynchronize();

}


__device__ float SiddonRayTracingTransposeAtomicAdd(float* pImg, float val, float3 src, float3 dst,
		int nx, int ny, int nz, float dx, float dy, float dz)
{
	SiddonTracingVars var = InitializeSiddon(src, dst, nx, ny, nz, dx, dy, dz);

	if (var.ng.x < 0 || var.ng.x >= nx || var.ng.y < 0 || var.ng.y >= ny || var.ng.z < 0 || var.ng.z >= nz)
	{
		// no intersections
		return 0;
	}

	int move = 0;
	bool isTracing = true;
	int nyz = ny * nz;
	pImg += var.ng.x * nyz + var.ng.y * nz + var.ng.z;

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
				move = nyz;
				if (var.ng.x >= nx) isTracing = false;
			}
			else
			{
				var.ng.x--;
				move = -nyz;
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
				move = nz;
				if (var.ng.y >= ny) isTracing = false;
			}
			else
			{
				var.ng.y--;
				move = -nz;
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
				move = 1;
				if (var.ng.z >= nz) isTracing = false;
			}
			else
			{
				var.ng.z--;
				move = -1;
				if (var.ng.z < 0) isTracing = false;
			}
		}

		atomicAdd(pImg, val * (var.alphaNow - var.alphaPrev));
		var.alphaPrev = var.alphaNow;
		pImg += move;
	}

	return var.rayLength;
}


__global__ void BackprojectionSiddonAtomicAddKernel(float* pImg, const float* pPrj,
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
	float z = (iv - (nv-1) / 2.0f + off_v) * dv;

	// src, dst points, and convert to image coordinate
	float3 src = make_float3(dso * sinDeg, -dso * cosDeg, z);
	float3 dstRel = make_float3(dsd * sin_a, -dso + dsd * cos_a, z);
	float3 dst = make_float3(dstRel.x * cosDeg - dstRel.y * sinDeg, dstRel.x * sinDeg + dstRel.y * cosDeg, z);

	SiddonRayTracingTransposeAtomicAdd(pImg, pPrj[iu * nview * nv + iview * nv + iv],
			src, dst, nx, ny, nz, dx, dy, dz);

}

void fan3D::BackprojectionSiddonAtomicAdd(float* pcuImg, const float* pcuPrj)
{
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), nv);

	BackprojectionSiddonAtomicAddKernel<<<blocks, threads>>>(pcuImg, pcuPrj, m_pcuDeg,
			nx, ny, nz, dx, dy, dz, nu, nview, nv, da, dv, off_a, off_v, dsd, dso);
	cudaDeviceSynchronize();

}

