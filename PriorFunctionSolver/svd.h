#pragma once

extern "C" int svdcmp_host(float* a, float* w, float* v, int m, int n);

__host__ __device__ int svdcmp(float* a, float* w, float* v, float* rv1, int m, int n);

__host__ __device__ void svdrecon(float* u, float* w, float* v, float* rv1, int m, int n);
