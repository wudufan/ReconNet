#include "fan.h"

#include <stdexcept>
#include <iostream>

using namespace std;


fan3D::fan3D()
{
	m_pcuImg = NULL;
	m_pcuPrj = NULL;
	m_pcuaImg = NULL;
	m_pcuaPrj = NULL;
	m_texImg = 0;
	m_texPrj = 0;
	m_pcuDeg = NULL;

	nx = 0;
	ny = 0;
	nz = 0;
	dx = 0;
	dy = 0;
	dz = 0;

	nu = 0;
	nview = 0;
	nv = 0;
	da = 0;
	dv = 0;
	off_a = 0;
	off_v = 0;

	dsd = 0;
	dso = 0;

}

fan3D::~fan3D()
{
	this->Destroy();
}

void fan3D::Setup(const float* pDeg, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso)
{
	this->nx = nx;
	this->ny = ny;
	this->nz = nz;
	this->dx = dx;
	this->dy = dy;
	this->dz = dz;

	this->nu = nu;
	this->nview = nview;
	this->nv = nv;
	this->da = da;
	this->dv = dv;
	this->off_a = off_a;
	this->off_v = off_v;

	this->dsd = dsd;
	this->dso = dso;

	// memory allocate
	try
	{
		// cuda memory

		if (cudaSuccess != cudaMalloc(&m_pcuDeg, sizeof(float) * nview))
		{
			throw std::runtime_error("Angle array allocation error");
		}
		cudaMemcpy(m_pcuDeg, pDeg, sizeof(float) * nview, cudaMemcpyHostToDevice);

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		// sinogram
		if (cudaSuccess != cudaMalloc3DArray(&m_pcuaPrj, &channelDesc, make_cudaExtent(nv, nview, nu)))
		{
			throw std::runtime_error("Projection array allocation failure!");
		}

		cudaResourceDesc resDescPrj;
		memset(&resDescPrj, 0, sizeof(resDescPrj));
		resDescPrj.resType = cudaResourceTypeArray;
		resDescPrj.res.array.array = m_pcuaPrj;

		if (cudaSuccess != cudaCreateTextureObject(&m_texPrj, &resDescPrj, &texDesc, NULL))
		{
			throw std::runtime_error("Projection texture binding failure!");
		}


		if (cudaSuccess != cudaMalloc(&m_pcuPrj, sizeof(float) * nu * nview * nv))
		{
			throw std::runtime_error("Projection allocation failure!");
		}
		if (cudaSuccess != cudaMalloc(&m_pcuImg, sizeof(float) * nx * ny * nz))
		{
			throw std::runtime_error("Image allocation failure!");
		}


		if (cudaSuccess != cudaMalloc3DArray(&m_pcuaImg, &channelDesc, make_cudaExtent(nz, ny, nx)))
		{
			throw std::runtime_error("Image array allocation failure!");
		}

		cudaResourceDesc resDescImg;
		memset(&resDescImg, 0, sizeof(resDescImg));
		resDescImg.resType = cudaResourceTypeArray;
		resDescImg.res.array.array = m_pcuaImg;

		if (cudaSuccess != cudaCreateTextureObject(&m_texImg, &resDescImg, &texDesc, NULL))
		{
			throw std::runtime_error("Image texture binding failure!");
		}

	}
	catch (std::exception &e)
	{
		cerr << e.what() << endl;
		this->Destroy();
		throw std::runtime_error(e.what());
	}

}

void fan3D::Destroy()
{
	if (m_pcuDeg != NULL)
	{
		cudaFree(m_pcuDeg);
		m_pcuDeg = NULL;
	}
	if (m_pcuImg != NULL)
	{
		cudaFree(m_pcuImg);
		m_pcuImg = NULL;
	}
	if (m_pcuPrj != NULL)
	{
		cudaFree(m_pcuPrj);
		m_pcuPrj = NULL;
	}
	if (m_texImg != 0)
	{
		cudaDestroyTextureObject(m_texImg);
		m_texImg = 0;
	}
	if (m_texPrj != 0)
	{
		cudaDestroyTextureObject(m_texPrj);
		m_texPrj = 0;
	}
	if (m_pcuaImg != NULL)
	{
		cudaFreeArray(m_pcuaImg);
		m_pcuaImg = NULL;
	}
	if (m_pcuaPrj != NULL)
	{
		cudaFreeArray(m_pcuaPrj);
		m_pcuaPrj = NULL;
	}


}

void fan3D::bindImgTex(float* pcuImg)
{
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr(pcuImg, nz * sizeof(float), nz, ny);
	copyParams.dstArray = m_pcuaImg;
	copyParams.extent = make_cudaExtent(nz, ny, nx);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);

}

void fan3D::bindPrjTex(float* pcuPrj)
{
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr(pcuPrj, nv * sizeof(float), nv, nview);
	copyParams.dstArray = m_pcuaPrj;
	copyParams.extent = make_cudaExtent(nv, nview, nu);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);

}
