#pragma once

#include <cuda_runtime.h>

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

struct Grid
{
	int nx;
	int ny;
	int nz;
	float dx;
	float dy;
	float dz;
	float cx;
	float cy;
	float cz;
};

struct Detector
{
	int nu;
	int nv;
	float du;
	float dv;
	float off_u;
	float off_v;
};

inline __device__ __host__ Grid MakeGrid(int nx, int ny, int nz, float dx, float dy, float dz,
		float cx, float cy, float cz)
{
	Grid grid;
	grid.nx = nx;
	grid.ny = ny;
	grid.nz = nz;
	grid.dx = dx;
	grid.dy = dy;
	grid.dz = dz;
	grid.cx = cx;
	grid.cy = cy;
	grid.cz = cz;

	return grid;
}

inline __device__ __host__ Detector MakeDetector(int nu, int nv, float du, float dv, float off_u, float off_v)
{
	Detector det;
	det.nu = nu;
	det.nv = nv;
	det.du = du;
	det.dv = dv;
	det.off_u = off_u;
	det.off_v = off_v;

	return det;
}

__device__ float SiddonRayTracing(float* pPrj, const float* pImg, float3 src, float3 dst,
		int nc, const Grid grid);

__device__ float SiddonRayTracingTransposeAtomicAdd(float* pImg, float val, float3 src, float3 dst,
		int nc, const Grid grid);

__device__ __host__ void MoveSourceDstNearGrid(float3& src, float3&dst, const Grid grid);


