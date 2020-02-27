#include "Projector.h"

#include <exception>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include <iostream>
#include <sstream>

using namespace std;

extern "C" int SetDevice(int device)
{
	return cudaSetDevice(device);
}

Projector::Projector()
{
	m_stream = NULL;

	nBatches = 0;
	nChannels = 0;

	nx = 0;
	ny = 0;
	nz = 0;
	dx = 0;
	dy = 0;
	dz = 0;
	cx = 0;
	cy = 0;
	cz = 0;

	nu = 0;
	nview = 0;
	nv = 0;
	du = 0;
	dv = 0;
	off_u = 0;
	off_v = 0;

	dsd = 0;
	dso = 0;

	typeProjector = 0;

}

Projector::~Projector()
{

}

void Projector::SetCudaStream(const cudaStream_t& stream)
{
	m_stream = stream;
}

void Projector::Setup(int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v,
		float dsd, float dso, int typeProjector)
{
	this->nBatches = nBatches;
	this->nChannels = nChannels;

	this->nx = nx;
	this->ny = ny;
	this->nz = nz;
	this->dx = dx;
	this->dy = dy;
	this->dz = dz;
	this->cx = 0;
	this->cy = 0;
	this->cz = 0;

	this->nu = nu;
	this->nview = nview;
	this->nv = nv;
	this->du = du;
	this->dv = dv;
	this->off_u = off_u;
	this->off_v = off_v;

	this->dsd = dsd;
	this->dso = dso;

	this->typeProjector = typeProjector;
}

void Projector::Setup(int nBatches, int nChannels,
		int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v,
		float dsd, float dso, int typeProjector)
{
	this->Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz, nu, nview, nv, du, dv,
			off_u, off_v, dsd, dso, typeProjector);

	this->cx = cx;
	this->cy = cy;
	this->cz = cz;
}

cFilter::cFilter(): Projector()
{
	typeFilter = 0;
}

cFilter::~cFilter()
{

}

void cFilter::Setup(int nBatches, int nChannels,int nu, int nview,
			int nv, float du, float dv, float off_u, float off_v,
			float dsd, float dso, int typeFilter, int typeProjector)
{
	this->nBatches = nBatches;
	this->nChannels = nChannels;

	this->nu = nu;
	this->nview = nview;
	this->nv = nv;
	this->du = du;
	this->dv = dv;
	this->off_u = off_u;
	this->off_v = off_v;

	this->dsd = dsd;
	this->dso = dso;

	this->typeProjector = typeProjector;
	this->typeFilter = typeFilter;
}

cProjector::cProjector(): Projector()
{
	m_pcuImg = NULL;
	m_pcuPrj = NULL;
	m_arrImg = NULL;
	m_arrPrj = NULL;
	m_texImg = 0;
	m_texPrj = 0;

}

void cProjector::DestroyTextures()
{
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
	if (m_arrImg != NULL)
	{
		cudaFreeArray(m_arrImg);
		m_arrImg = NULL;
	}
	if (m_arrPrj != NULL)
	{
		cudaFreeArray(m_arrPrj);
		m_arrPrj = NULL;
	}

}

cProjector::~cProjector()
{
	DestroyTextures();
}

void cProjector::SetupImgTexture()
{
	try
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		if (cudaSuccess != cudaMalloc(&m_pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw std::runtime_error("Image allocation failure!");
		}

		if (cudaSuccess != cudaMalloc3DArray(&m_arrImg, &channelDesc, make_cudaExtent(nz, ny, nx)))
		{
			throw std::runtime_error("Image array allocation failure!");
		}

		cudaResourceDesc resDescImg;
		memset(&resDescImg, 0, sizeof(resDescImg));
		resDescImg.resType = cudaResourceTypeArray;
		resDescImg.res.array.array = m_arrImg;

		if (cudaSuccess != cudaCreateTextureObject(&m_texImg, &resDescImg, &texDesc, NULL))
		{
			throw std::runtime_error("Image texture binding failure!");
		}

	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cProjector::SetupImgTexture() failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw std::runtime_error(oss.str().c_str());
	}
}

void cProjector::SetupPrjTexture()
{
	try
	{
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
		if (cudaSuccess != cudaMalloc3DArray(&m_arrPrj, &channelDesc, make_cudaExtent(nv, nview, nu)))
		{
			throw std::runtime_error("Projection array allocation failure!");
		}

		cudaResourceDesc resDescPrj;
		memset(&resDescPrj, 0, sizeof(resDescPrj));
		resDescPrj.resType = cudaResourceTypeArray;
		resDescPrj.res.array.array = m_arrPrj;

		if (cudaSuccess != cudaCreateTextureObject(&m_texPrj, &resDescPrj, &texDesc, NULL))
		{
			throw std::runtime_error("Projection texture binding failure!");
		}

		if (cudaSuccess != cudaMalloc(&m_pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw std::runtime_error("Projection allocation failure!");
		}

	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cProjector::SetupPrjTexture() failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw std::runtime_error(oss.str().c_str());
	}
}

void cProjector::BindImgTex(float* pcuImg)
{
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr(pcuImg, nChannels * nz * sizeof(float), nz, ny);
	copyParams.dstArray = m_arrImg;
	copyParams.extent = make_cudaExtent(nz, ny, nx);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3DAsync(&copyParams, m_stream);
}

void cProjector::BindPrjTex(float* pcuPrj)
{
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr(pcuPrj, nChannels * nv * sizeof(float), nv, nview);
	copyParams.dstArray = m_arrPrj;
	copyParams.extent = make_cudaExtent(nv, nview, nu);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3DAsync(&copyParams, m_stream);
}
