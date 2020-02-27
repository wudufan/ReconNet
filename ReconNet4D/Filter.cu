#include "Projector.h"
#include "cudaMath.h"

#include <cufft.h>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cuda_runtime_api.h>

using namespace std;

static const int filterHamming = 1;
static const int filterHann = 2;
static const int filterCosine = 3;

__global__ void ComplexMultiply2D(cufftComplex* res, const cufftComplex* op1, const cufftComplex* op2,
		int nx, int ny)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	int ind = iy * nx + ix;
	cufftComplex val1 = op1[ind];
	cufftComplex val2 = op2[ind];
	cufftComplex val;

	val.x = val1.x * val2.x - val1.y * val2.y;
	val.y = val1.x * val2.y + val1.y * val2.x;

	res[ind] = val;
}

__global__ void CopyPrjToPad(float* pcuPrjPad, const float* pcuPrj, int iv,
		int nu, int nuPad, int nview, int nv, int nc)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iview = blockIdx.y * blockDim.y + threadIdx.y;

	if (iu >= nu || iview >= nview)
	{
		return;
	}

	pcuPrjPad[iview * nuPad + iu] = pcuPrj[iu * nv * nview * nc + iview * nv * nc + iv * nc];
}

__global__ void CopyPadToPrj(const float* pcuPrjPad, float* pcuPrj, int iv,
		int nu, int nuPad, int nview, int nv, int nc)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iview = blockIdx.y * blockDim.y + threadIdx.y;

	if (iu >= nu || iview >= nview)
	{
		return;
	}

	pcuPrj[iu * nv * nview * nc + iview * nv * nc + iv * nc] = pcuPrjPad[iview * nuPad + iu];
}

void getRamp(cufftComplex* pcuFreqKernel, int nu, int nview, float da, int filterType, cudaStream_t& stream,
		bool isEqualSpace = false)
{
	int filterLen = 2 * nu - 1;

	// fft plan
	cufftHandle plan;
	if (CUFFT_SUCCESS != cufftPlan1d(&plan, filterLen, CUFFT_C2C, 1))
	{
		throw std::runtime_error("cufftPlan1d failure in getRampEA");
	}

	if (CUFFT_SUCCESS != cufftSetStream(plan, stream))
	{
		throw std::runtime_error("cudaSetStream failure in getRampEA");
	}

	// RL kernel
	cufftComplex* pcuRamp = NULL;
	if (cudaSuccess != cudaMalloc(&pcuRamp, sizeof(cufftComplex) * filterLen))
	{
		throw std::runtime_error("pcuRamp allocation error in getRampEA");
	}
	cufftComplex* pRamp = new cufftComplex [filterLen];
	if (isEqualSpace)
	{
		// equispace
		for (int i = 0; i < filterLen; i++)
		{
			int k = i - (nu - 1);
			if (k == 0)
			{
				pRamp[i].x = 1 / (4 * da * da);
			}
			else if (k % 2 != 0)
			{
				pRamp[i].x = -1 / (PI * PI * k * k * da * da);
			}
			else
			{
				pRamp[i].x = 0;
			}
			pRamp[i].y = 0;
		}
	}
	else
	{
		// equiangular
		for (int i = 0; i < filterLen; i++)
		{
			int k = i - (nu - 1);
			if (k == 0)
			{
				pRamp[i].x = 1 / (4 * da * da);
			}
			else if (k % 2 != 0)
			{
				pRamp[i].x = -1 / (PI * PI * sinf(k*da) * sinf(k*da));
			}
			else
			{
				pRamp[i].x = 0;
			}
			pRamp[i].y = 0;
		}
	}

	cudaMemcpyAsync(pcuRamp, pRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyHostToDevice, stream);
	cufftExecC2C(plan, pcuRamp, pcuRamp, CUFFT_FORWARD);
	cudaMemcpyAsync(pRamp, pcuRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyDeviceToHost, stream);

	// weighting window in frequency domain
	cufftComplex* pWindow = new cufftComplex [filterLen];
	switch(filterType)
	{
	case filterHamming:
		// Hamming
		for (int i = 0; i < filterLen; i++)
		{
			pWindow[i].x = 0.54f + 0.46f * cosf(2 * PI * i / (float)(filterLen-1));
			pWindow[i].y = 0;
		}
		break;
	case filterHann:
		for (int i = 0; i < filterLen; i++)
		{
			pWindow[i].x = 0.5f + 0.5f * cosf(2 * PI * i / (float)(filterLen-1));
			pWindow[i].y = 0;
		}
		break;
	case filterCosine:
		for (int i = 0; i < filterLen; i++)
		{
			pWindow[i].x = abs(cosf(PI * i / (float)(filterLen-1)));
			pWindow[i].y = 0;
		}
		break;
	default:
		for (int i = 0; i < filterLen; i++)
		{
			pWindow[i].x = 1;
			pWindow[i].y = 0;
		}
	}

	// Apply window on the filter
	for (int i = 0; i < filterLen; i++)
	{
		float real = pRamp[i].x * pWindow[i].x - pRamp[i].y * pWindow[i].y;
		float imag = pRamp[i].x * pWindow[i].y + pRamp[i].y * pWindow[i].x;
		pRamp[i].x = real;
		pRamp[i].y = imag;
	}

	cudaMemcpyAsync(pcuRamp, pRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyHostToDevice, stream);
	cufftExecC2C(plan, pcuRamp, pcuRamp, CUFFT_INVERSE);
	cudaMemcpyAsync(pRamp, pcuRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyDeviceToHost, stream);

	cufftReal *pcuRealKernel = NULL;
	if (cudaSuccess != cudaMalloc(&pcuRealKernel, sizeof(cufftReal) * filterLen))
	{
		throw std::runtime_error("pRealKernel allocation error in getRampEA");
	}
	cufftReal* pRealKernel = new cufftReal [filterLen];
	for (int i = 0; i < filterLen; i++)
	{
		pRealKernel[i] = pRamp[i].x / filterLen;
	}
	cudaMemcpyAsync(pcuRealKernel, pRealKernel, sizeof(cufftReal) * filterLen, cudaMemcpyHostToDevice, stream);

	cufftHandle planR2C;
	cufftPlan1d(&planR2C, filterLen, CUFFT_R2C, 1);
	cufftSetStream(planR2C, stream);
	cufftExecR2C(planR2C, pcuRealKernel, pcuFreqKernel);
	for (int i = 1; i < nview; i++)
	{
		cudaMemcpyAsync(pcuFreqKernel + i * nu, pcuFreqKernel, sizeof(cufftComplex) * nu,
				cudaMemcpyDeviceToDevice, stream);
	}

	delete [] pRamp;
	delete [] pWindow;
	delete [] pRealKernel;
	cudaFree(pcuRamp);
	cudaFree(pcuRealKernel);
	cufftDestroy(plan);
	cufftDestroy(planR2C);

}

void cFilterFan::Filter(float* pcuFPrj, const float* pcuPrj)
{
	bool isEqualSpace = false;

	// the filter is carried out for each different v
	int filterLen = nu * 2 - 1;

	// projection
	float* pcuPrjPad = NULL;
	if (cudaSuccess != cudaMalloc(&pcuPrjPad, sizeof(float) * filterLen * nview))
	{
		throw std::runtime_error("pcuPrjPad allocation failure in fan3D::Filter");
	}

	// freq projection
	cufftComplex* pcuFreqPrj = NULL;
	if (cudaSuccess != cudaMalloc(&pcuFreqPrj, sizeof(cufftComplex) * nu * nview))
	{
		throw std::runtime_error("pcuFreqPrj allocation failure in fan3D::Filter");
	}

	// filter
	cufftComplex* pcuFilter = NULL;
	if (cudaSuccess != cudaMalloc(&pcuFilter, sizeof(cufftComplex) * nu * nview))
	{
		throw std::runtime_error("pcuFilter allocation failure in fan3D::Filter");
	}
	getRamp(pcuFilter, nu, nview, du, typeFilter, m_stream, isEqualSpace);

	// Get projection weighting
	float* pw = new float [nu];
	if (isEqualSpace)
	{
		for (int i = 0; i < nu; i++)
		{
			float u = ((i - (nu - 1) / 2.f) - off_u) * du;
			pw[i] = dsd / sqrtf(dsd * dsd + u * u);
		}
	}
	else
	{
		for (int i = 0; i < nu; i++)
		{
			float angle = ((i - (nu - 1) / 2.f) - off_u) * du;
			pw[i] = cosf(angle);
		}
	}

	float* pcuw = NULL;
	if (cudaSuccess != cudaMalloc(&pcuw, sizeof(float) * nu * nview))
	{
		throw std::runtime_error("pcuw allocation failure in fan3D::Filter");
	}
	cudaMemcpyAsync(pcuw, pw, sizeof(float) * nu, cudaMemcpyHostToDevice, m_stream);
	for (int i = 1; i < nview; i++)
	{
		cudaMemcpyAsync(pcuw + i * nu, pcuw, sizeof(float) * nu, cudaMemcpyDeviceToDevice, m_stream);
	}
	delete [] pw;

	// fft plan
	cufftHandle plan;
	if (CUFFT_SUCCESS != cufftPlanMany(&plan, 1, &filterLen, NULL, 1, filterLen, NULL, 1, nu, CUFFT_R2C, nview))
	{
		throw std::runtime_error("fft plan error in fan3D::Filter");
	}
	cufftSetStream(plan, m_stream);

	cufftHandle planInverse;
	if (CUFFT_SUCCESS != cufftPlanMany(&planInverse, 1, &filterLen, NULL, 1, nu, NULL, 1, filterLen, CUFFT_C2R, nview))
	{
		throw std::runtime_error("ifft plan error in fan3D::Filter");
	}
	cufftSetStream(plan, m_stream);

	// kernel threads and blocks
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), 1);
	float scale;
	if (isEqualSpace)
	{
		scale = PI / nview * du * dsd / dso / filterLen;
	}
	else
	{
		scale = PI / nview * du / dso / filterLen;
	}

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int iv = 0; iv < nv; iv++)
		{
			for (int ic = 0; ic < nChannels; ic++)
			{
				cudaMemsetAsync(pcuPrjPad, 0, sizeof(float) * filterLen * nview, m_stream);
				CopyPrjToPad<<<blocks, threads, 0, m_stream>>>(pcuPrjPad,
						pcuPrj + ib * nu * nview * nv * nChannels + ic,
						iv, nu, filterLen, nview, nv, nChannels);

				// pre weighting
				Multiply2D<<<blocks, threads, 0, m_stream>>>(pcuPrjPad, pcuPrjPad, pcuw,
						nu, nview, filterLen, filterLen, nu);
				cudaDeviceSynchronize();

				cufftExecR2C(plan, pcuPrjPad, pcuFreqPrj);
				ComplexMultiply2D<<<blocks, threads, 0, m_stream>>>(pcuFreqPrj, pcuFreqPrj, pcuFilter, nu, nview);
				cudaDeviceSynchronize();
				cufftExecC2R(planInverse, pcuFreqPrj, pcuPrjPad);

				// post scaling
				Scale2D<<<blocks, threads, 0, m_stream>>>(pcuPrjPad + nu - 1, pcuPrjPad + nu - 1,
						scale, nu, nview, filterLen, filterLen);

				CopyPadToPrj<<<blocks, threads, 0, m_stream>>>(pcuPrjPad + nu - 1,
						pcuFPrj + ib * nu * nview * nv * nChannels + ic,
						iv, nu, filterLen, nview, nv, nChannels);
			}
		}
	}

	cufftDestroy(plan);
	cufftDestroy(planInverse);
	cudaFree(pcuPrjPad);
	cudaFree(pcuFreqPrj);
	cudaFree(pcuFilter);
	cudaFree(pcuw);
}

void cFilterParallel::Filter(float* pcuFPrj, const float* pcuPrj)
{
	bool isEqualSpace = true;

	// the filter is carried out for each different v
	int filterLen = nu * 2 - 1;

	// projection
	float* pcuPrjPad = NULL;
	if (cudaSuccess != cudaMalloc(&pcuPrjPad, sizeof(float) * filterLen * nview))
	{
		throw std::runtime_error("pcuPrjPad allocation failure in fan3D::Filter");
	}

	// freq projection
	cufftComplex* pcuFreqPrj = NULL;
	if (cudaSuccess != cudaMalloc(&pcuFreqPrj, sizeof(cufftComplex) * nu * nview))
	{
		throw std::runtime_error("pcuFreqPrj allocation failure in fan3D::Filter");
	}

	// filter
	cufftComplex* pcuFilter = NULL;
	if (cudaSuccess != cudaMalloc(&pcuFilter, sizeof(cufftComplex) * nu * nview))
	{
		throw std::runtime_error("pcuFilter allocation failure in fan3D::Filter");
	}
	getRamp(pcuFilter, nu, nview, du, typeFilter, m_stream, isEqualSpace);

	// no weighting for parallel filtering

	// fft plan
	cufftHandle plan;
	if (CUFFT_SUCCESS != cufftPlanMany(&plan, 1, &filterLen, NULL, 1, filterLen, NULL, 1, nu, CUFFT_R2C, nview))
	{
		throw std::runtime_error("fft plan error in fan3D::Filter");
	}
	cufftSetStream(plan, m_stream);

	cufftHandle planInverse;
	if (CUFFT_SUCCESS != cufftPlanMany(&planInverse, 1, &filterLen, NULL, 1, nu, NULL, 1, filterLen, CUFFT_C2R, nview))
	{
		throw std::runtime_error("ifft plan error in fan3D::Filter");
	}
	cufftSetStream(plan, m_stream);

	// kernel threads and blocks
	dim3 threads(32, 32, 1);
	dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), 1);
	float scale = PI / nview * du / filterLen;

	for (int ib = 0; ib < nBatches; ib++)
	{
		for (int iv = 0; iv < nv; iv++)
		{
			for (int ic = 0; ic < nChannels; ic++)
			{
				cudaMemsetAsync(pcuPrjPad, 0, sizeof(float) * filterLen * nview, m_stream);
				CopyPrjToPad<<<blocks, threads, 0, m_stream>>>(pcuPrjPad,
						pcuPrj + ib * nu * nview * nv * nChannels + ic,
						iv, nu, filterLen, nview, nv, nChannels);

				// no pre weighting for parallel filtering
				cudaDeviceSynchronize();

				cufftExecR2C(plan, pcuPrjPad, pcuFreqPrj);
				ComplexMultiply2D<<<blocks, threads, 0, m_stream>>>(pcuFreqPrj, pcuFreqPrj, pcuFilter, nu, nview);
				cudaDeviceSynchronize();
				cufftExecC2R(planInverse, pcuFreqPrj, pcuPrjPad);

				// post scaling
				Scale2D<<<blocks, threads, 0, m_stream>>>(pcuPrjPad + nu - 1, pcuPrjPad + nu - 1,
						scale, nu, nview, filterLen, filterLen);

				CopyPadToPrj<<<blocks, threads, 0, m_stream>>>(pcuPrjPad + nu - 1,
						pcuFPrj + ib * nu * nview * nv * nChannels + ic,
						iv, nu, filterLen, nview, nv, nChannels);
			}
		}
	}

	cufftDestroy(plan);
	cufftDestroy(planInverse);
	cudaFree(pcuPrjPad);
	cudaFree(pcuFreqPrj);
	cudaFree(pcuFilter);
}

extern "C" void cFilterFanFilter(float* pFPrj, const float* pPrj,
		int nBatches, int nChannels,int nu, int nview,
		int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, int typeFilter = 0, int typeProjector = 0)
{
	cFilterFan filter;
	filter.Setup(nBatches, nChannels, nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeFilter, typeProjector);
	float* pcuFPrj = NULL;
	float* pcuPrj = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuFPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw runtime_error("pcuFPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw runtime_error("pcuPrj allocation failed");
		}
		cudaMemset(pcuFPrj, 0, sizeof(float) * nBatches * nu * nview * nv * nChannels);
		cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);

		filter.Filter(pcuFPrj, pcuPrj);
	}
	catch (exception &e)
	{
		if (pcuFPrj != NULL) cudaFree(pcuFPrj);
		if (pcuPrj != NULL) cudaFree(pcuPrj);

		ostringstream oss;
		oss << "cFilterFanFilter() failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	cudaMemcpy(pFPrj, pcuFPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuFPrj);
	cudaFree(pcuPrj);

}

extern "C" void cFilterParallelFilter(float* pFPrj, const float* pPrj,
		int nBatches, int nChannels,int nu, int nview,
		int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, int typeFilter = 0, int typeProjector = 0)
{
	cFilterParallel filter;
	filter.Setup(nBatches, nChannels, nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeFilter, typeProjector);
	float* pcuFPrj = NULL;
	float* pcuPrj = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuFPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw runtime_error("pcuFPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw runtime_error("pcuPrj allocation failed");
		}
		cudaMemset(pcuFPrj, 0, sizeof(float) * nBatches * nu * nview * nv * nChannels);
		cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);

		filter.Filter(pcuFPrj, pcuPrj);
	}
	catch (exception &e)
	{
		if (pcuFPrj != NULL) cudaFree(pcuFPrj);
		if (pcuPrj != NULL) cudaFree(pcuPrj);

		ostringstream oss;
		oss << "cFilterFanFilter() failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	cudaMemcpy(pFPrj, pcuFPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuFPrj);
	cudaFree(pcuPrj);

}
