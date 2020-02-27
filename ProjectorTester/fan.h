#pragma once

#include <cuda_runtime.h>

typedef cudaStream_t cudaStream_t;

#define PI 3.141592657f

class fan3D
{

public:
	// image parameters
	int nx;
	int ny;
	int nz;
	float dx;
	float dy;
	float dz;

	// sinogram parameters
	int nu;
	int nview;
	int nv;
	float da;
	float dv;
	float off_a;
	float off_v;

	// geometries
	float dsd;
	float dso;


public:
	void Setup(const float* pDeg, int nx, int ny, int nz, float dx, float dy, float dz,
			int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
			float dsd, float dso);

public:
	void Destroy();

public:
	// time: 0.0482s; initialization time: 0.0011s;
	// software interpolation but texture memory access: 0.0968s
	void ProjectionRasterTexture(float* pcuPrj, const float* pcuImg);

	// time: 0.333s; without boundary check: 0.61s (slower due to memory access)
	void ProjectionRaster(float* pcuPrj, const float* pcuImg);

	// time: 0.131s; initialization time: 0.0013s
	void ProjectionSiddon(float* pcuPrj, const float* pcuImg);

	// time: 0.0767s
	void ProjectionSiddonTexture(float* pcuPrj, const float* pcuImg);

public:
	// time: 0.12493s
	void BackprojectionPixelTexture(float* pcuImg, const float* pcuPrj);

	// time: 0.3452s
	void BackprojectionRayTextureAtomicAdd(float* pcuImg, const float* pcuPrj);

	// time: 0.105122s
	void BackprojectionSiddonAtomicAdd(float* pcuImg, const float* pcuPrj);

	// no need to test, given the good performance of atomicAdd
	void BackprojectionSiddon(float* pcuImg, const float* pcuPrj);

protected:
	float* m_pcuImg;
	float* m_pcuPrj;
	float* m_pcuDeg;
	cudaArray* m_pcuaImg;
	cudaArray* m_pcuaPrj;
	cudaTextureObject_t m_texImg;
	cudaTextureObject_t m_texPrj;

public:
	fan3D();
	virtual ~fan3D();

protected:
	void bindImgTex(float* pcuImg);
	void bindPrjTex(float* pcuPrj);

};
