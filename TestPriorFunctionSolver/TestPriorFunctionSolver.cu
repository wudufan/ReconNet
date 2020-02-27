#include <iostream>
#include "../PriorFunctionSolver/svd.h"

using namespace std;

int main(int argc, char** argv)
{
	const int m = 4;
	const int n = 3;

	float a[] = {
			0.9572, 0.4218, 0.6557,
		    0.4854, 0.9157, 0.0357,
		    0.8003, 0.7922, 0.8491,
		    0.1419, 0.9595, 0.9340};
	float* w = new float [n];
	float* v = new float [n * n];

	int res = svdcmp_host(a, w, v, m, n);

	cout << res << endl;

	cout << "U = " << endl;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << a[i * n + j] << ", ";
		}
		cout << endl;
	}

	cout << "W = " << endl;
	for (int i = 0; i < n; i++)
	{
		cout << w[i] << ", ";
	}
	cout << endl;

	cout << "V = " << endl;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << v[i * n + j] << ", ";
		}
		cout << endl;
	}

	delete [] w;
	delete [] v;

	return 0;
}
