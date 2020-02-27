#include "svd.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

static const float eps = 1e-6f;

__host__ __device__ float shrink(float x, float mu, float p)
{
	float absx = fabs(x);
	if (x < eps)
	{
		return 0;
	}

	float y = absx - mu * powf(absx, p - 1);
	y = y > 0 ? y : 0;
	return x > 0 ? y : -y;
}

// calculate a = u*shrink_p(w, mu)*vT
// a is mxn, w is length n, v is nxn, rv1 is length n
// s is the w before shrink, used to calculate the loss function
__host__ __device__ int NuclearNormSoftThresh(float* a, float* w, float* s, float* v, float* rv1, int m, int n, float mu, float p)
{
	// svd decomposition
	int res = svdcmp(a, s, v, rv1, m, n);
	if (res < 0)
	{
		return res;
	}

	// shrink w
	for (int i = 0; i < n; i++)
	{
		w[i] = shrink(s[i], mu, p);
	}

	// svd reconstruction
	svdrecon(a, w, v, rv1, m, n);

	return 0;
}

extern "C" int NuclearNormSoftThresh_host(float* a, float* s, int m, int n, float mu, float p)
{
	float* w = new float [n];
	float* v = new float [n * n];
	float* rv1 = new float [n];

	int res = NuclearNormSoftThresh(a, w, s, v, rv1, m, n, mu, p);

	delete [] w;
	delete [] v;
	delete [] rv1;

	return res;
}

// compute multiple nuclear norm soft threshold simultaneously
// a is batchxmxn, w is batchxn, v is batchxnxn, rv1 is batchxn
// outputs are stored in a
__global__ void NuclearNormSoftThreshKernel(float* a, float* w, float* s, float* v, float* rv1, int m, int n, int batch, float mu, float p)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= batch)
	{
		return;
	}

	a += id * m * n;
	w += id * n;
	s += id * n;
	v += id * n * n;
	rv1 += id * n;

	NuclearNormSoftThresh(a, w, s, v, rv1, m, n, mu, p);

}

void NuclearNormSoftThresh_gpu(float* a, float* s, int m, int n, int batch, float mu, float p)
{
	float* w = NULL;
	float* v = NULL;
	float* rv1 = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&w, batch * n * sizeof(float)))
		{
			throw runtime_error("w allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&v, batch * n * n * sizeof(float)))
		{
			throw runtime_error("v allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&rv1, batch * n * sizeof(float)))
		{
			throw runtime_error("rv1 allocation failed");
		}

		dim3 threads(256, 1, 1);
		dim3 blocks(ceilf(batch / (float)threads.x), 1, 1);
		NuclearNormSoftThreshKernel<<<blocks, threads>>>(a, w, s, v, rv1, m, n, batch, mu, p);
	}
	catch (exception& e)
	{
		if (w != NULL) cudaFree(w);
		if (v != NULL) cudaFree(v);
		if (rv1 != NULL) cudaFree(rv1);

		ostringstream oss;
		oss << "NuclearNormSoftThresh_gpu failed" << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	cudaFree(w);
	cudaFree(v);
	cudaFree(rv1);
}

extern "C" void cNuclearNormSoftThresh(float* a, float* s, int m, int n, int batch, float mu, float p)
{
	float* pcua = NULL;
	float* pcus = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcua, sizeof(float) * batch * m * n))
		{
			throw runtime_error("pcua allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcus, sizeof(float) * batch * n))
		{
			throw runtime_error("pcus allocation failed");
		}

		cudaMemcpy(pcua, a, sizeof(float) * batch * m * n, cudaMemcpyHostToDevice);

		NuclearNormSoftThresh_gpu(pcua, pcus, m, n, batch, mu, p);

		cudaMemcpy(a, pcua, sizeof(float) * batch * m * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(s, pcus, sizeof(float) * batch * n, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		if (pcua != NULL) cudaFree(pcua);
		if (pcus != NULL) cudaFree(pcus);

		ostringstream oss;
		oss << "cNuclearNormSoftThresh failed: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	cudaFree(pcua);
	cudaFree(pcus);
}



