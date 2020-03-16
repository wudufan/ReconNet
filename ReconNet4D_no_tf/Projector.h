#pragma once

#include <stdio.h>

#define PI 3.141592657f

class Projector
{
public:
	int nBatches;		// number of batches
	int nChannels;		// number of channels

	// image parameters
	int nx;
	int ny;
	int nz;
	float dx; // mm
	float dy; // mm
	float dz; // mm
	float cx; // center of image in mm
	float cy;
	float cz;

	// sinogram parameters
	int nu;
	int nview;
	int nv;
	float du;
	float dv;
	float off_u;
	float off_v;

	// geometries
	float dsd;
	float dso;

	int typeProjector; // to tag slight changes between different versions, e.g. BP for FBP do not need length factor


public:
	void Setup(int nBatches, int nChannels,
			int nx, int ny, int nz, float dx, float dy, float dz,
			int nu, int nview, int nv, float du, float dv, float off_u, float off_v,
			float dsd, float dso, int typeProjector = 0);

	void Setup(int nBatches, int nChannels,
			int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
			int nu, int nview, int nv, float du, float dv, float off_u, float off_v,
			float dsd, float dso, int typeProjector = 0);

public:
	void SetCudaStream(const cudaStream_t& stream);

public:
	virtual void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) {};
	virtual void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) {};

protected:
	cudaStream_t m_stream;

public:
	Projector();
	virtual ~Projector();

};

// projector with own memory management, potential collision with Tensorflow
class cProjector: public Projector
{
protected:
	float* m_pcuImg;
	float* m_pcuPrj;
	cudaArray_t m_arrImg;
	cudaArray_t m_arrPrj;
	cudaTextureObject_t m_texImg;
	cudaTextureObject_t m_texPrj;

public:
	cProjector();
	virtual ~cProjector();

public:
	void SetupImgTexture();
	void BindImgTex(float* pcuImg);
	void SetupPrjTexture();
	void BindPrjTex(float* pcuPrj);
	void DestroyTextures();

};

class cFilter: public Projector
{
public:
	int typeFilter;

public:
	cFilter();
	virtual ~cFilter();

public:
	void Setup(int nBatches, int nChannels,int nu, int nview,
			int nv, float du, float dv, float off_u, float off_v,
			float dsd, float dso, int typeFilter = 0, int typeProjector = 0);

public:
	virtual void Filter(float* pcuFPrj, const float* pcuPrj) {};
};

class SiddonFan: public Projector
{
public:
	SiddonFan(): Projector() {}
	~SiddonFan() {}

public:
	void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) override;
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};

class SiddonParallel: public Projector
{
public:
	SiddonParallel(): Projector() {}
	~SiddonParallel() {}

public:
	void ProjectionParallel(const float* pcuImg, float* pcuPrj, const float* pcuDetCenter,
			const float* pcuDetU, const float* pcuDetV, const float* pcuInvRayDir);
	void BackprojectionParallel(float* pcuImg, const float* pcuPrj, const float* pcuDetCenter,
			const float* pcuDetU, const float* pcuDetV, const float* pcuInvRayDir);

};

class SiddonCone: public Projector
{
public:
	SiddonCone(): Projector() {}
	~SiddonCone() {}

public:
	void ProjectionAbitrary(const float* pcuImg, float* pcuPrj, const float* pcuDetCenter,
			const float* pcuDetU, const float* pcuDetV, const float* pcuSrc);
	void BackprojectionAbitrary(float* pcuImg, const float* pcuPrj, const float* pcuDetCenter,
			const float* pcuDetU, const float* pcuDetV, const float* pcuSrc);

};

class cPixelDrivenFan: public cProjector
{
public:
	cPixelDrivenFan(): cProjector() {}
	~cPixelDrivenFan() {}

public:
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};


class cFilterFan: public cFilter
{
public:
	cFilterFan(): cFilter() {}
	~cFilterFan() {}

public:
	void Filter(float* pcuFPrj, const float* pcuPrj) override;
};

class cFilterParallel: public cFilter
{
public:
	cFilterParallel(): cFilter() {}
	~cFilterParallel() {}

public:
	void Filter(float* pcuFPrj, const float* pcuPrj) override;
};

