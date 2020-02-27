#include "cudaMath.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

__global__ void TVSQSKernel(float* grad, float* norm, float* var, const float* img, int nx, int ny)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	int ind = ix * ny + iy;
	float x = img[ind];
	float dx0, dx1, dy0, dy1;

	if (ix == 0)
	{
		dx0 = x;
	}
	else
	{
		dx0 = x - img[ind - ny];
	}

	if (ix == nx - 1)
	{
		dx1 = x;
	}
	else
	{
		dx1 = x - img[ind + ny];
	}

	if (iy == 0)
	{
		dy0 = x;
	}
	else
	{
		dy0 = x - img[ind - 1];
	}

	if (iy == ny - 1)
	{
		dy1 = x;
	}
	else
	{
		dy1 = x - img[ind + 1];
	}

	const float eps = 1e-6;
	float tvx0 = sqrtf(dx0 * dx0 + eps);
	float tvx1 = sqrtf(dx1 * dx1 + eps);
	float tvy0 = sqrtf(dy0 * dy0 + eps);
	float tvy1 = sqrtf(dy1 * dy1 + eps);

	grad[ind] = (dx0 / tvx0 + dx1 / tvx1 + dy0 / tvy0 + dy1 / tvy1) / 2;
	norm[ind] = 1 / tvx0 + 1 / tvx1 + 1 / tvy0 + 1 / tvy1;
	var[ind] = (tvx0 + tvx1 + tvy0 + tvy1) / 2;
}

extern "C" float cTVSQS2D(float* grad, float* norm, const float* img, int nx, int ny)
{
	float* pcuImg = NULL;
	float* pcuGrad = NULL;
	float* pcuNorm = NULL;
	float* pcuVar = NULL;


	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nx * ny))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuGrad, sizeof(float) * nx * ny))
		{
			throw runtime_error("pcuGrad allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuNorm, sizeof(float) * nx * ny))
		{
			throw runtime_error("pcuNorm allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuVar, sizeof(float) * nx * ny))
		{
			throw runtime_error("pcuVar allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
		cudaMemset(pcuGrad, 0, sizeof(float) * nx * ny);
		cudaMemset(pcuNorm, 0, sizeof(float) * nx * ny);
		cudaMemset(pcuVar, 0, sizeof(float) * nx * ny);
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuGrad != NULL) cudaFree(pcuGrad);
		if (pcuNorm != NULL) cudaFree(pcuNorm);
		if (pcuVar != NULL) cudaFree(pcuVar);

		ostringstream oss;
		oss << "cSiddonFanProjection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	dim3 threads(32,32,1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), 1);

	TVSQSKernel<<<blocks, threads>>>(pcuGrad, pcuNorm, pcuVar, pcuImg, nx, ny);

	cudaMemcpy(grad, pcuGrad, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
	cudaMemcpy(norm, pcuNorm, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);

	float* var = new float [nx * ny];
	cudaMemcpy(var, pcuVar, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
	float tv = 0;
	for (int i = 0; i < nx * ny; i++)
	{
		tv += var[i];
	}
	delete [] var;

	cudaFree(pcuImg);
	cudaFree(pcuGrad);
	cudaFree(pcuNorm);
	cudaFree(pcuVar);

	return tv;
}



