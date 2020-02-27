// currently some of the memories were allocated by cuda, such as textures
// memories allocated by streamExecutor may override the cuda memories, lead to potential memory leak
// currently allocating all the streamExecutor memories before any cuda memories works fine
// will remove any cuda memory allocating in the future

#define EIGEN_USE_GPU

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h>

#include <iostream>
#include <vector>
#include <string>

#include <time.h>

#include <cuda_runtime.h>
#include "Projector.h"

using namespace std;
using namespace tensorflow;
using ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("Projection4D")
	.Attr("output_shape: list(int) >= 3")
//	.Attr("voxel_sz: list(float) = [1, 1, 1]")
	.Attr("det_sz: list(float) = [0.0011844, 1.0]")
	.Attr("det_off: list(float) = [0, 0]")
	.Attr("dsd: float = 1085.6")
	.Attr("dso: float = 595")
	.Attr("type_projector: int = 0")
	.Input("img: float")
	.Input("angles: float")
	.Input("voxel_sz: float")
	.Output("sino: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context)
	{
		vector<int> sino_shape;
		TF_RETURN_IF_ERROR( context->GetAttr("output_shape", &sino_shape));

		std::vector<DimensionHandle> output_dim;
		output_dim.push_back(context->Dim(context->input(0), 0)); // batch_size
		for (int i = 0; i < 3; i++)
		{
			output_dim.push_back(context->MakeDim(sino_shape[i]));	// nu, nview, nv
		}
		output_dim.push_back(context->Dim(context->input(0), 4)); // number of channels

		context->set_output(0, context->MakeShape(output_dim));
		return Status::OK();
	});

REGISTER_OP("Backprojection4D")
	.Attr("output_shape: list(int) >= 3")
//	.Attr("voxel_sz: list(float) = [1, 1, 1]")
	.Attr("det_sz: list(float) = [0.0011844, 1.0]")
	.Attr("det_off: list(float) = [0, 0]")
	.Attr("dsd: float = 1085.6")
	.Attr("dso: float = 595")
	.Attr("type_projector: int = 0")
	.Input("sino: float")
	.Input("angles: float")
	.Input("voxel_sz: float")
	.Output("img: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context)
	{
		vector<int> img_shape;
		TF_RETURN_IF_ERROR( context->GetAttr("output_shape", &img_shape));

		std::vector<DimensionHandle> output_dim;
		output_dim.push_back(context->Dim(context->input(0), 0)); // batch_size
		for (int i = 0; i < 3; i++)
		{
			output_dim.push_back(context->MakeDim(img_shape[i]));	// nu, nview, nv
		}
		output_dim.push_back(context->Dim(context->input(0), 4)); // number of channels

		context->set_output(0, context->MakeShape(output_dim));
		return Status::OK();
	});

class NotImplementedOp: public OpKernel
{
public:
	explicit NotImplementedOp(OpKernelConstruction* context): OpKernel(context) {}

	void Compute(OpKernelContext* context) override
	{
		context->SetStatus(errors::Unimplemented("Reconstruction operation for CPU is not implemented"));
	}
};

class ReconNetOpBase : public OpKernel
{
public:
	explicit ReconNetOpBase(OpKernelConstruction* context): OpKernel(context)
	{
		pProjector = NULL;

		OP_REQUIRES_OK(context, context->GetAttr("output_shape", &outputShape));
//		OP_REQUIRES_OK(context, context->GetAttr("voxel_sz", &voxelSz));
		OP_REQUIRES_OK(context, context->GetAttr("det_sz", &detSz));
		OP_REQUIRES_OK(context, context->GetAttr("det_off", &detOff));
		OP_REQUIRES_OK(context, context->GetAttr("dsd", &dsd));
		OP_REQUIRES_OK(context, context->GetAttr("dso", &dso));
		OP_REQUIRES_OK(context, context->GetAttr("type_projector", &typeProjector));

		// TODO: positivity check
//		OP_REQUIRES(context, voxelSz.size() == 3, errors::InvalidArgument("Voxel size must have 3 dimensions"));
		OP_REQUIRES(context, detSz.size() == 2, errors::InvalidArgument("Detector size must have 2 dimensions"));
		OP_REQUIRES(context, detOff.size() == 2, errors::InvalidArgument("Detector offset must have 2 dimensions"));
	}

	virtual ~ReconNetOpBase()
	{
		if (pProjector != NULL)
		{
			delete pProjector;
		}
	}

protected:
	void ProjectorSetup(OpKernelContext* context, const TensorShape& imgShape, const TensorShape& prjShape)
	{
		if (pProjector == NULL)
		{
			// allocate projector according to typeProjector
			pProjector = new SiddonFan;
		}

		pProjector->SetCudaStream(context->eigen_gpu_device().stream());

		pProjector->Setup(imgShape.dim_size(0), imgShape.dim_size(4),
				imgShape.dim_size(1), imgShape.dim_size(2), imgShape.dim_size(3),
				voxelSz[0], voxelSz[1], voxelSz[2],
				prjShape.dim_size(1), prjShape.dim_size(2), prjShape.dim_size(3),
				detSz[0], detSz[1], detOff[0], detOff[1],
				dsd, dso, typeProjector);
	}

	void GetVoxelSz(OpKernelContext* context, int inputInd = 2)
	{
		const Tensor& voxelSzTensor = context->input(inputInd);

		OP_REQUIRES(context, voxelSzTensor.dims() == 1,
				errors::InvalidArgument("voxel_sz must have 1 dimension"));
		OP_REQUIRES(context, voxelSzTensor.dim_size(0) == 3,
				errors::InvalidArgument("voxel_sz.shape[0] must euqal to 3"));

		// get voxel size
		voxelSz = vector<float>(3);
		cudaMemcpyAsync(&voxelSz[0], voxelSzTensor.flat<float>().data(), sizeof(float) * 3,
				cudaMemcpyDeviceToHost, context->eigen_gpu_device().stream());
	}

protected:
	Projector* pProjector;
	vector<int> outputShape;
	vector<float> voxelSz;
	vector<float> detSz;
	vector<float> detOff;
	float dsd;
	float dso;
	int typeProjector;

};

class Projection4DOp: public ReconNetOpBase
{
public:
	explicit Projection4DOp(OpKernelConstruction* context): ReconNetOpBase(context)
	{

	}

	void Compute(OpKernelContext* context) override
	{
		// input tensor
		const Tensor& imgTensor = context->input(0);
		const Tensor& anglesTensor = context->input(1);

		OP_REQUIRES(context, imgTensor.dims() == 5,
				errors::InvalidArgument("Image must have 5 dimensions"));
		OP_REQUIRES(context, anglesTensor.dims() == 1,
				errors::InvalidArgument("angles must have 1 dimension"));
		OP_REQUIRES(context, anglesTensor.dim_size(0) == outputShape[1],
				errors::InvalidArgument("angles.shape[0] must equal to output_shape[1]"));

		// output tensor
		Tensor* prjTensor = NULL;
		TensorShape output_dim;
		output_dim.AddDim(imgTensor.dim_size(0));
		output_dim.AddDim(outputShape[0]);
		output_dim.AddDim(outputShape[1]);
		output_dim.AddDim(outputShape[2]);
		output_dim.AddDim(imgTensor.dim_size(4));
		OP_REQUIRES_OK(context, context->allocate_output(0, output_dim, &prjTensor));

		GetVoxelSz(context, 2);

		// computation
		const TensorShape& imgShape = imgTensor.shape();
		const TensorShape& prjShape = prjTensor->shape();

		ProjectorSetup(context, imgShape, prjShape);

		cudaMemset(prjTensor->flat<float>().data(), 0, sizeof(float) * prjTensor->NumElements());
		pProjector->Projection(imgTensor.flat<float>().data(), prjTensor->flat<float>().data(),
				anglesTensor.flat<float>().data());
	}

};

class Backprojection4DOp: public ReconNetOpBase
{
public:
	explicit Backprojection4DOp(OpKernelConstruction* context): ReconNetOpBase(context)
	{

	}

	void Compute(OpKernelContext* context) override
	{
		// input tensor
		const Tensor& prjTensor = context->input(0);
		const Tensor& anglesTensor = context->input(1);

		OP_REQUIRES(context, prjTensor.dims() == 5,
				errors::InvalidArgument("sino must have 5 dimensions"));
		OP_REQUIRES(context, anglesTensor.dims() == 1,
				errors::InvalidArgument("angles must have 1 dimension"));
		OP_REQUIRES(context, anglesTensor.dim_size(0) == prjTensor.dim_size(2),
				errors::InvalidArgument("angles.shape[0] must equal to sino.shape[2]"));

		// output tensor
		Tensor* imgTensor = NULL;
		TensorShape output_dim;
		output_dim.AddDim(prjTensor.dim_size(0));
		output_dim.AddDim(outputShape[0]);
		output_dim.AddDim(outputShape[1]);
		output_dim.AddDim(outputShape[2]);
		output_dim.AddDim(prjTensor.dim_size(4));
		OP_REQUIRES_OK(context, context->allocate_output(0, output_dim, &imgTensor));

		GetVoxelSz(context, 2);

		// computation
		const TensorShape& imgShape = imgTensor->shape();
		const TensorShape& prjShape = prjTensor.shape();

		ProjectorSetup(context, imgShape, prjShape);

		cudaMemset(imgTensor->flat<float>().data(), 0, sizeof(float) * imgTensor->NumElements());
		pProjector->Backprojection(imgTensor->flat<float>().data(), prjTensor.flat<float>().data(),
				anglesTensor.flat<float>().data());

	}
};

REGISTER_KERNEL_BUILDER(Name("Projection4D").Device(DEVICE_GPU), Projection4DOp);
REGISTER_KERNEL_BUILDER(Name("Projection4D").Device(DEVICE_CPU), NotImplementedOp);

REGISTER_KERNEL_BUILDER(Name("Backprojection4D").Device(DEVICE_GPU), Backprojection4DOp);
REGISTER_KERNEL_BUILDER(Name("Backprojection4D").Device(DEVICE_CPU), NotImplementedOp);



REGISTER_OP("ProjectionConeAbitrary4D")
	.Attr("output_shape: list(int) >= 3")
	.Attr("det_sz: list(float) = [1.0, 1.0]")
	.Attr("det_off: list(float) = [0, 0]")
	.Attr("voxel_sz: list(float) = [1.0, 1.0, 1.0]")
	.Attr("voxel_center: list(float) = [0.0, 0.0, 0.0]")
	.Input("img: float")
	.Input("geometry: float")
	.Output("prj: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context)
	{
		vector<int> prj_shape;
		TF_RETURN_IF_ERROR( context->GetAttr("output_shape", &prj_shape));

		std::vector<DimensionHandle> output_dim;
		output_dim.push_back(context->Dim(context->input(0), 0)); // batch_size
		for (int i = 0; i < 3; i++)
		{
			output_dim.push_back(context->MakeDim(prj_shape[i]));	// nu, nview, nv
		}
		output_dim.push_back(context->Dim(context->input(0), 4)); // number of channels

		context->set_output(0, context->MakeShape(output_dim));
		return Status::OK();
	});


REGISTER_OP("BackprojectionConeAbitrary4D")
	.Attr("output_shape: list(int) >= 3")
	.Attr("det_sz: list(float) = [1.0, 1.0]")
	.Attr("det_off: list(float) = [0, 0]")
	.Attr("voxel_sz: list(float) = [1.0, 1.0, 1.0]")
	.Attr("voxel_center: list(float) = [0.0, 0.0, 0.0]")
	.Input("prj: float")
	.Input("geometry: float")
	.Output("img: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context)
	{
		vector<int> img_shape;
		TF_RETURN_IF_ERROR( context->GetAttr("output_shape", &img_shape));

		std::vector<DimensionHandle> output_dim;
		output_dim.push_back(context->Dim(context->input(0), 0)); // batch_size
		for (int i = 0; i < 3; i++)
		{
			output_dim.push_back(context->MakeDim(img_shape[i]));	// nx, ny, nz
		}
		output_dim.push_back(context->Dim(context->input(0), 4)); // number of channels

		context->set_output(0, context->MakeShape(output_dim));
		return Status::OK();
	});

class ProjectionConeAbitraryBase : public OpKernel
{
public:
	explicit ProjectionConeAbitraryBase(OpKernelConstruction* context): OpKernel(context)
	{
		pProjector = NULL;

		OP_REQUIRES_OK(context, context->GetAttr("output_shape", &outputShape));
		OP_REQUIRES_OK(context, context->GetAttr("voxel_sz", &voxelSz));
		OP_REQUIRES_OK(context, context->GetAttr("voxel_center", &voxelCenter));
		OP_REQUIRES_OK(context, context->GetAttr("det_sz", &detSz));
		OP_REQUIRES_OK(context, context->GetAttr("det_off", &detOff));

		// TODO: positivity check
		OP_REQUIRES(context, voxelSz.size() == 3, errors::InvalidArgument("Voxel size must have 3 dimensions"));
		OP_REQUIRES(context, voxelCenter.size() == 3, errors::InvalidArgument("Voxel center must have 3 dimensions"));
		OP_REQUIRES(context, detSz.size() == 2, errors::InvalidArgument("Detector size must have 2 dimensions"));
		OP_REQUIRES(context, detOff.size() == 2, errors::InvalidArgument("Detector offset must have 2 dimensions"));
	}

	virtual ~ProjectionConeAbitraryBase()
	{
		if (pProjector != NULL)
		{
			delete pProjector;
		}
	}

protected:
	void ProjectorSetup(OpKernelContext* context, const TensorShape& imgShape, const TensorShape& prjShape)
	{
		if (pProjector == NULL)
		{
			// allocate projector according to typeProjector
			pProjector = new SiddonCone;
		}

		pProjector->SetCudaStream(context->eigen_gpu_device().stream());

		pProjector->Setup(imgShape.dim_size(0), imgShape.dim_size(4),
				imgShape.dim_size(1), imgShape.dim_size(2), imgShape.dim_size(3),
				voxelSz[0], voxelSz[1], voxelSz[2],
				voxelCenter[0], voxelCenter[1], voxelCenter[2],
				prjShape.dim_size(1), prjShape.dim_size(2), prjShape.dim_size(3),
				detSz[0], detSz[1], detOff[0], detOff[1],
				0, 0, 1);
	}

protected:
	Projector* pProjector;
	vector<int> outputShape;
	vector<float> voxelSz;
	vector<float> voxelCenter;
	vector<float> detSz;
	vector<float> detOff;

};

class ProjectionConeAbitrary4DOp: public ProjectionConeAbitraryBase
{
public:
	explicit ProjectionConeAbitrary4DOp(OpKernelConstruction* context): ProjectionConeAbitraryBase(context)
	{

	}

	void Compute(OpKernelContext* context) override
	{
		// input tensor
		const Tensor& imgTensor = context->input(0);
		const Tensor& geoTensor = context->input(1);

		OP_REQUIRES(context, imgTensor.dims() == 5,
				errors::InvalidArgument("Image must have 5 dimensions"));
		OP_REQUIRES(context, geoTensor.dims() == 3,
				errors::InvalidArgument("geometry must have 3 dimensions"));
		OP_REQUIRES(context, geoTensor.dim_size(0) == 4,
				errors::InvalidArgument("geometry.shape[0] must equal to 4 (det_center, det_u, det_v, src)"));
		OP_REQUIRES(context, geoTensor.dim_size(1) == outputShape[1],
				errors::InvalidArgument("geometry.shape[1] must equal to outputShape[1] (nPrjs)"));
		OP_REQUIRES(context, geoTensor.dim_size(2) == 3,
				errors::InvalidArgument("geometry.shape[2] must equal to 3"));

		// output tensor
		Tensor* prjTensor = NULL;
		TensorShape output_dim;
		output_dim.AddDim(imgTensor.dim_size(0));
		output_dim.AddDim(outputShape[0]);
		output_dim.AddDim(outputShape[1]);
		output_dim.AddDim(outputShape[2]);
		output_dim.AddDim(imgTensor.dim_size(4));
		OP_REQUIRES_OK(context, context->allocate_output(0, output_dim, &prjTensor));

		// computation
		const TensorShape& imgShape = imgTensor.shape();
		const TensorShape& prjShape = prjTensor->shape();

		ProjectorSetup(context, imgShape, prjShape);

		cudaMemset(prjTensor->flat<float>().data(), 0, sizeof(float) * prjTensor->NumElements());
		int stride = geoTensor.dim_size(1) * geoTensor.dim_size(2);
		const float* pGeo = geoTensor.flat<float>().data();
		((SiddonCone*)pProjector)->ProjectionAbitrary(imgTensor.flat<float>().data(), prjTensor->flat<float>().data(),
				pGeo, pGeo + stride, pGeo + 2 * stride, pGeo + 3 * stride);
	}

};


class BackprojectionConeAbitrary4DOp: public ProjectionConeAbitraryBase
{
public:
	explicit BackprojectionConeAbitrary4DOp(OpKernelConstruction* context): ProjectionConeAbitraryBase(context)
	{

	}

	void Compute(OpKernelContext* context) override
	{
		// input tensor
		const Tensor& prjTensor = context->input(0);
		const Tensor& geoTensor = context->input(1);

		OP_REQUIRES(context, prjTensor.dims() == 5,
				errors::InvalidArgument("Image must have 5 dimensions"));
		OP_REQUIRES(context, geoTensor.dims() == 3,
				errors::InvalidArgument("geometry must have 3 dimensions"));
		OP_REQUIRES(context, geoTensor.dim_size(0) == 4,
				errors::InvalidArgument("geometry.shape[0] must equal to 4 (det_center, det_u, det_v, src)"));
		OP_REQUIRES(context, geoTensor.dim_size(1) == prjTensor.dim_size(2),
				errors::InvalidArgument("geometry.shape[1] must equal to prj.shape[2] (nPrjs)"));
		OP_REQUIRES(context, geoTensor.dim_size(2) == 3,
				errors::InvalidArgument("geometry.shape[2] must equal to 3"));

		// output tensor
		Tensor* imgTensor = NULL;
		TensorShape output_dim;
		output_dim.AddDim(prjTensor.dim_size(0));
		output_dim.AddDim(outputShape[0]);
		output_dim.AddDim(outputShape[1]);
		output_dim.AddDim(outputShape[2]);
		output_dim.AddDim(prjTensor.dim_size(4));
		OP_REQUIRES_OK(context, context->allocate_output(0, output_dim, &imgTensor));

		// computation
		const TensorShape& imgShape = imgTensor->shape();
		const TensorShape& prjShape = prjTensor.shape();

		ProjectorSetup(context, imgShape, prjShape);

		cudaMemset(imgTensor->flat<float>().data(), 0, sizeof(float) * imgTensor->NumElements());
		int stride = geoTensor.dim_size(1) * geoTensor.dim_size(2);
		const float* pGeo = geoTensor.flat<float>().data();
		((SiddonCone*)pProjector)->BackprojectionAbitrary(imgTensor->flat<float>().data(), prjTensor.flat<float>().data(),
				pGeo, pGeo + stride, pGeo + 2 * stride, pGeo + 3 * stride);
	}

};

REGISTER_KERNEL_BUILDER(Name("ProjectionConeAbitrary4D").Device(DEVICE_GPU), ProjectionConeAbitrary4DOp);
REGISTER_KERNEL_BUILDER(Name("ProjectionConeAbitrary4D").Device(DEVICE_CPU), NotImplementedOp);

REGISTER_KERNEL_BUILDER(Name("BackprojectionConeAbitrary4D").Device(DEVICE_GPU), BackprojectionConeAbitrary4DOp);
REGISTER_KERNEL_BUILDER(Name("BackprojectionConeAbitrary4D").Device(DEVICE_CPU), NotImplementedOp);








