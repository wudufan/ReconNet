#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include "fan.h"
#include <cstdlib>
#include <cstring>

#include <cuda_runtime_api.h>

using namespace std;

void ReadImg(float* img, const char* filename, int nx, int ny, int nz)
{
	short* simg = new short [nx * ny * nz];

	ifstream ifs(filename, ios::binary);
	ifs.read((char*)simg, sizeof(short) * nx * ny * nz);
	ifs.close();

	for (int iz = 0; iz < nz; iz++)
	{
		for (int iy = 0; iy < ny; iy++)
		{
			for (int ix = 0; ix < nx; ix++)
			{
				img[ix * ny * nz + iy * nz + iz] =
						((float)simg[iz * nx * ny + iy * nx + ix] + 1000.f) / 1000.f * 0.019f;
			}
		}
	}

	delete [] simg;
}

void RandomImg(float* img, int nx, int ny, int nz, unsigned int seed = 0)
{
	srand(seed);
	for (int i = 0; i < nx * ny * nz; i++)
	{
		img[i] = rand() / (float)RAND_MAX;
	}
}

void WriteImg(float* img, const char* filename, int nx, int ny, int nz)
{
	float* wimg = new float [nx * ny * nz];

	for (int iz = 0; iz < nz; iz++)
	{
		for (int iy = 0; iy < ny; iy++)
		{
			for (int ix = 0; ix < nx; ix++)
			{
				wimg[iz * nx * ny + iy * nx + ix] = img[ix * ny * nz + iy * nz + iz];
			}
		}
	}

	ofstream ofs(filename, ios::binary);
	ofs.write((char*)wimg, sizeof(float) * nx * ny * nz);
	ofs.close();

	delete [] wimg;
}

double dot(float* img1, float* img2, int nx, int ny, int nz)
{
	double val = 0;
	for (int i = 0; i < nx * ny * nz; i++)
	{
		val += img1[i] * img2[i];
	}

	return val;
}

int main(int argc, char** argv)
{
	cudaSetDevice(1);

	int nx = 512;
	int ny = 512;
	int nz = 1;
	float dx = 1;
	float dy = 1;
	float dz = 1;
	int nu = 1024;
	int nview = 2304;
	int nv = 1;
	float dsd = 1086.5f;
	float dso = 595.f;
	float da = 1.2858f / dsd;
	float dv = 1;
	float off_a = -1.125;
	float off_v = 0;
	int nRepeatFP = 1;
	int nRepeatBP = 1;

	float* angles = new float [nview];
	for (int i = 0; i < nview; i++)
	{
		angles[i] = 2 * PI / nview * i;
	}

	fan3D fan3d;
	fan3d.Setup(angles, nx, ny, nz, dx, dy, dz, nu, nview, nv, da, dv, off_a, off_v, dsd, dso);

	const char* filename = "/home/dufan/workspace/Reconstruction/DeepRecon/train/data/noiseless/img/L067.raw";
	float* img = new float [nx * ny * nz];
	float* bp = new float [nx * ny * nz];
	float* prj = new float [nu * nview * nv];
	float* fp = new float [nu * nview * nv];
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	cudaMalloc(&pcuImg, sizeof(float) * nx * ny * nz);
	cudaMalloc(&pcuPrj, sizeof(float) * nu * nview * nv);



	// projection
	ReadImg(img, filename, nx, ny, nz);
	RandomImg(img, nx, ny, nz, 0);
	cudaMemcpy(pcuImg, img, sizeof(float) * nx * ny * nz, cudaMemcpyHostToDevice);

	cudaMemset(pcuPrj, 0, sizeof(float) * nu * nview * nv);
	clock_t start = clock();
	for (int i = 0; i < nRepeatFP; i++)
	{
//		fan3d.ProjectionRasterTexture(pcuPrj, pcuImg);
//		fan3d.ProjectionRaster(pcuPrj, pcuImg);
//		fan3d.ProjectionSiddon(pcuPrj, pcuImg);
		fan3d.ProjectionSiddonTexture(pcuPrj, pcuImg);
	}
	cudaDeviceSynchronize();
	clock_t end = clock();
	float sec = (end - start) / (float)CLOCKS_PER_SEC;
	cout << "Elapsed time for projection = " << sec << "s" << endl;

	cudaMemcpy(fp, pcuPrj, sizeof(float) * nu * nview * nv, cudaMemcpyDeviceToHost);
	for (int i = 0; i < nu * nview * nv; i++)
	{
		fp[i] /= nRepeatFP;
	}
	// end of projection



	// backprojection
	memcpy(prj, fp, sizeof(float) * nu * nview * nv);
	RandomImg(prj, nu, nview, nv, 1);
	cudaMemcpy(pcuPrj, prj, sizeof(float) * nu * nview * nv, cudaMemcpyHostToDevice);

	cudaMemset(pcuImg, 0, sizeof(float) * nx * ny * nz);
	start = clock();
	for (int i = 0; i < nRepeatBP; i++)
	{
//		fan3d.BackprojectionPixelTexture(pcuImg, pcuPrj);
//		fan3d.BackprojectionRayTextureAtomicAdd(pcuImg, pcuPrj);
		fan3d.BackprojectionSiddonAtomicAdd(pcuImg, pcuPrj);
	}
	cudaDeviceSynchronize();
	end = clock();
	sec = (end - start) / (float)CLOCKS_PER_SEC;
	cout << "Elapsed time for backprojection = " << sec << "s" << endl;

	cudaMemcpy(bp, pcuImg, sizeof(float) * nx * ny * nz, cudaMemcpyDeviceToHost);
	for (int i = 0; i < nx * ny * nz; i++)
	{
		bp[i] /= nRepeatBP;
	}
	// end of backprojection

	// dot product
	double val1 = dot(fp, prj, nu, nview, nv);
	double val2 = dot(bp, img, nx, ny, nz);
	double ratio = val1 / val2;
	cout << "p'Ax / x'A'p = " << ratio << endl;
//	cout << val1 << ", " << val2 << endl;

//	ostringstream oss;
//	oss << "fpSiddonTex-" << nu << "-" << nview << "-" << nv << ".raw";
//	WriteImg(fp, oss.str().c_str(), nu, nview, nv);
//
//	oss.str("");
//	oss.clear();
//	oss << "bpSiddonAtomic-" << nx << "-" << ny << "-" << nz << ".raw";
//	WriteImg(bp, oss.str().c_str(), nx, ny, nz);

	fan3d.Destroy();

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	delete [] angles;
	delete [] img;
	delete [] prj;
	delete [] fp;
	delete [] bp;

	return 0;
}
