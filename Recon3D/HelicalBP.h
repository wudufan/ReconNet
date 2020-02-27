#include "Projector.h"

class cHelicalBPFromParallelConebeam: public cProjector
{
public:
	int nviewPerPI;	// number of projections per PI segment
	float theta0; 	// start theta angle for the projections
	float volZ0;	// starting slice z position, projection start z assumed 0
	float zrot;		// increase in z per rotation (2PI)
	int mPI;		// the BP will look within [theta - mPI * PI, theta + mPI * PI]
	float Q;		// smoothing parameter for weighting


public:
	cHelicalBPFromParallelConebeam();
	~cHelicalBPFromParallelConebeam() {}

public:
	void Setup(int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
			int nu, int nview, int nv, float du, float dv, float off_u, float off_v, float dsd, float dso,
			int nviewPerPI, float theta0, float volZ0, float zrot, int mPI, float Q);

public:
	// pDeg is not used here
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pDeg) override;

	// debugging function
	void BackprojectionParallel2D(float* pcuImg, const float* pcuPrj, const float* pDeg);
};
