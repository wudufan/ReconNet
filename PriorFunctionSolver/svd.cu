
#include "svd.h"

#include <cmath>

float __host__ __device__ SQR(float x)
{
	return x * x;
}

float __host__ __device__ SIGN(float a, float b)
{
	return b >= 0.0 ? fabs(a) : -fabs(a);
}

float __host__ __device__ FMAX(float a, float b)
{
	return a > b ? a : b;
}

float __host__ __device__ IMIN(int a, int b)
{
	return a < b ? a : b;
}

// Computes (a^2 + b^2)1/2 without destructive underflow or overflow.
// Numerical Recipes in C, 2nd edition
__host__ __device__ float pythag(float a, float b)
{
	float absa,absb;
	absa = fabs(a);
	absb = fabs(b);
	if (absa > absb) return absa * sqrtf(1.0f + SQR(absb/absa));
	else return (absb == 0.0f ? 0.0f : absb * sqrtf(1.0f + SQR(absa/absb)));
}

// Given a matrix a of size mxn, this routine computes its singular value decomposition,
// A = U·W·V T. U replace A on output. The diagonal matrix of singular
// values W is output as a vector w of size n. The matrix V (not the transpose V T ) is output as v
// of size nxn.
// rv1 is an auxiliary vector with length n
__host__ __device__ int svdcmp(float* a, float* w, float* v, float* rv1, int m, int n)
{
	int flag, nm, l;
	float anorm, c, f, g, h, s, scale, x, y, z;

	g = scale = anorm = 0.0f; // Householder reduction to bidiagonal form.
	for (int i = 0; i < n; i++)
	{
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0f;
		if (i < m)
		{
			for (int k = i; k < m; k++)
			{
				scale += fabs(a[k * n + i]);
			}
			if (scale)
			{
				for (int k = i; k < m; k++)
				{
					a[k * n + i] /= scale;
					s += a[k * n + i] * a[k * n + i];
				}
				f = a[i * n + i];
				g = -SIGN(sqrtf(s),f);
				h = f * g - s;
				a[i * n + i] = f - g;
				for (int j = l; j < n; j++)
				{
					s = 0.0f;
					for (int k = i; k < m; k++)
					{
						s += a[k * n + i] * a[k * n + j];
					}
					f = s / h;
					for (int k = i; k < m; k++)
					{
						a[k * n + j] += f * a[k * n + i];
					}
				}
				for (int k = i; k < m; k++)
				{
					a[k * n + i] *= scale;
				}
			}
		}
		w[i] = scale * g;
		g = s = scale = 0.0f;
		if (i < m && i != n-1)
		{
			for (int k = l; k < n; k++)
			{
				scale += fabs(a[i * n + k]);
			}
			if (scale)
			{
				for (int k = l; k < n; k++)
				{
					a[i * n + k] /= scale;
					s += a[i * n + k] * a[i * n + k];
				}
				f = a[i * n + l];
				g = -SIGN(sqrtf(s), f);
				h = f * g - s;
				a[i * n + l] = f - g;
				for (int k = l; k < n; k++)
				{
					rv1[k] = a[i * n + k] / h;
				}
				for (int j = l; j < m; j++)
				{
					s = 0.0f;
					for (int k = l; k < n; k++)
					{
						s += a[j * n + k] * a[i * n + k];
					}
					for (int k = l; k < n; k++)
					{
						a[j * n + k] += s * rv1[k];
					}
				}
				for (int k = l; k < n; k++)
				{
					a[i * n + k] *= scale;
				}
			}
		}
		anorm = FMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}
	for (int i = n - 1; i >= 0; i--) // Accumulation of right-hand transformations.
	{
		if (i < n - 1)
		{
			if (g)
			{
				for (int j = l;j < n; j++) //Double division to avoid possible underflow.
				{
					v[j * n + i] = (a[i * n + j] / a[i * n + l]) / g;
				}
				for (int j = l; j < n; j++)
				{
					s = 0.0f;
					for (int k = l; k < n; k++)
					{
						s += a[i * n + k] * v[k * n + j];
					}
					for (int k = l; k < n; k++)
					{
						v[k * n + j] += s * v[k * n + i];
					}
				}
			}
			for (int j = l; j < n; j++)
			{
				v[i * n + j] = v[j * n + i] = 0.0f;
			}
		}
		v[i * n + i] = 1.0f;
		g = rv1[i];
		l = i;
	}
	for (int i = IMIN(m, n) - 1; i >= 0; i--)  // Accumulation of left-hand transformations.
	{
		l = i + 1;
		g = w[i];
		for (int j = l; j < n; j++)
		{
			a[i * n + j] = 0.0f;
		}
		if (g)
		{
			g = 1.0f / g;
			for (int j = l; j < n; j++)
			{
				s = 0.0f;
				for (int k = l; k < m; k++)
				{
					s += a[k * n + i] * a[k * n + j];
				}
				f = (s / a[i * n + i]) * g;
				for (int k = i; k < m; k++)
				{
					a[k * n + j] += f * a[k * n + i];
				}
			}
			for (int j = i; j < m; j++)
			{
				a[j * n + i] *= g;
			}
		}
		else
		{
			for (int j = i; j < m; j++)
			{
				a[j * n + i] = 0.0f;
			}
		}
		a[i * n + i] += 1;
	}
	// Diagonalization of the bidiagonal form: Loop over singular values,
	// and over allowed iterations.
	for (int k = n - 1; k >= 0; k--)
	{
		for (int its = 0; its < 30; its++)
		{
			flag = 1;
			for (l = k; l >= 0; l--)  // Test for splitting. Note that rv1[0] is always zero.
			{
				nm = l - 1;
				if ((float)(fabs(rv1[l]) + anorm) == anorm)
				{
					flag = 0;
					break;
				}
				if ((float)(fabs(w[nm]) + anorm) == anorm)
				{
					break;
				}
			}
			if (flag)
			{
				c = 0.0f;  // Cancellation of rv1[l], if l > 0.
				s = 1.0f;
				for (int i = l; i <= k; i++)
				{
					f = s * rv1[i];
					rv1[i] = c * rv1[i];
					if ((float)(fabs(f) + anorm) == anorm)
					{
						break;
					}
					g = w[i];
					h = pythag(f, g);
					w[i] = h;
					h = 1.0f / h;
					c = g * h;
					s = -f * h;
					for (int j = 0; j < m; j++)
					{
						y = a[j * n + nm];
						z = a[j * n + i];
						a[j * n + nm] = y * c + z * s;
						a[j * n + i] = z * c - y * s;
					}
				}
			}
			z = w[k];
			if (l == k)  // Convergence.
			{
				if (z < 0.0f)  // Singular value is made nonnegative.
				{
					w[k] = -z;
					for (int j = 0; j < n; j++)
					{
						v[j * n + k] = -v[j * n + k];
					}
				}
				break;
			}
			if (its == 29)
			{
				return -1;
			}
			x = w[l]; // Shift from bottom 2-by-2 minor.
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = pythag(f, 1.0f);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
			c = s = 1.0f; // Next QR transformation:
			for (int j = l; j <= nm; j++) {
				int i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = pythag(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y *= c;
				for (int jj = 0; jj < n; jj++)
				{
					x = v[jj * n + j];
					z = v[jj * n + i];
					v[jj * n + j] = x * c + z * s;
					v[jj * n + i] = z * c - x * s;
				}
				z = pythag(f, h);
				w[j] = z; // Rotation can be arbitrary if z = 0.
				if (z)
				{
					z = 1.0f / z;
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;
				for (int jj = 0; jj < m; jj++)
				{
					y = a[jj * n + j];
					z = a[jj * n + i];
					a[jj * n + j] = y * c + z * s;
					a[jj * n + i] = z * c - y * s;
				}
			}
			rv1[l] = 0.0f;
			rv1[k] = f;
			w[k]=x;
		}
	}

	return 0;
}

// construct A = U W VT, A replace U on output
// U is mxn, W is length n, V is nxn. A is mxn
// rv1 is an auxiliary vector of length n to hold every row of u
__host__ __device__ void svdrecon(float* u, float* w, float* v, float* rv1, int m, int n)
{
	// compute u * w
	for (int i = 0; i < n; i++)
	{
		float s = w[i];
		for (int j = 0; j < m; j++)
		{
			u[j * n + i] *= s;
		}
	}

	// compute u * w * vt
	for (int i = 0; i < m; i++)
	{
		// hold the current row of u
		for (int j = 0; j < n; j++)
		{
			rv1[j] = u[i * n + j];
		}

		for (int j = 0; j < n; j++)
		{
			float s = 0;
			for (int k = 0; k < n; k++)
			{
				s += rv1[k] * v[j * n + k];
			}
			u[i * n + j] = s;
		}
	}

}

extern "C" int svdcmp_host(float* a, float* w, float* v, int m, int n)
{
	float* rv1 = new float [n];

	int res = svdcmp(a, w, v, rv1, m, n);

	delete [] rv1;

	return res;

}

extern "C" void svdrecon_host(float* u, float* w, float* v, int m, int n)
{
	float* rv1 = new float [n];
	svdrecon(u, w, v, rv1, m, n);
	delete [] rv1;
}
