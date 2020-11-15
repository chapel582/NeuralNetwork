#ifndef NEURAL_NET_H

#include "project_flags.h"
#include "int_shuffler.h"
#include "matrix.h"

// TODO: Need to have a platform independent way of handling threads and file io
#include <windows.h>

struct dense_layer
{
	/* NOTE: 
	Inputs matrix to this layer has dimensions K x N where K is the number of 
	samples in the batch and N is the number of dimensions for the output of the 
	previous layer
	Therefore, Weights has dimensions N x M, M is the dimension of the output 
	for this dense layer
	Bias has dimensions 1 x M
	*/
	matrix Weights;
	matrix Bias;
};

struct softmax_layer
{
	matrix Intermediate;
};

struct dense_layer_train_data
{
	matrix WeightsDelta;
	matrix BiasDelta;
	matrix LayerGradient;
	float LearningRate;
};

struct relu_train_data
{
	matrix LayerGradient;
};

struct mse_train_data
{
	matrix LayerGradient;
};

struct softmax_xentropy_train_data
{
	matrix LayerGradient;
};

typedef enum
{
	LayerType_Dense,
	LayerType_Relu,
	LayerType_Softmax,
	LayerType_CrossEntropy,
	LayerType_SoftmaxCrossEntropy,
	LayerType_Mse,
	LayerType_Count // NOTE: doubles as invalid term!
} layer_type;

struct layer_link;
struct layer_link
{
	layer_type Type;
	void* Data;
	void* Output;
	layer_link* Next;
	layer_link* Previous;
};

struct matrix_op_args
{
	float Float;
	matrix* M1;
	matrix* M2;
	matrix* Result;
	int Start;
	int Stride;
};
struct matrix_op_jobs
{
	uint32_t NumThreads;
	matrix_op_args* Args;
	HANDLE* Handles; // TODO: make this platform independent
};

struct neural_net
{
	uint32_t NumLayers;
	uint32_t BatchSize;
	uint32_t InputDim;
	layer_link* FirstLink;
	layer_link* LastLink;

	// NOTE: op jobs if needed 
	matrix_op_jobs* MatrixOpJobs;
};

struct neural_net_trainer
{
	neural_net* NeuralNet;
	void** TrainDataArray;
	matrix* MiniBatchData;
	matrix* MiniBatchLabels;
};

#pragma pack(push, 1)
struct model_header
{
	uint32_t NumLayers;
	uint32_t InputDim;
};

struct layer_header
{
	layer_type Type;
};
#pragma pack(pop)

void InitDenseLayers(neural_net* NeuralNet);
void AllocMatrixOpJobs(matrix_op_jobs** Result, uint32_t NumThreads);
float MseForward(
	matrix_op_jobs* MatrixOpJobs, matrix* Predictions, matrix* Labels
);
void AddMeanSquared(neural_net* NeuralNet);
void MeanSquaredBack(
	matrix_op_jobs* MatrixOpJobs,
	matrix* Predictions, 
	matrix* Labels, 
	mse_train_data* TrainData
);

HOST_PREFIX DEVICE_PREFIX
void ReluForwardCore(
	matrix* M1, matrix* Result, uint32_t Start, uint32_t Stride
);
HOST_PREFIX DEVICE_PREFIX
void ReluBackCore(
	matrix* Inputs,
	matrix* NextLayerGradient,
	matrix* LayerGradient,
	uint32_t Start,
	uint32_t Stride
);
HOST_PREFIX DEVICE_PREFIX
void SoftmaxForwardCore(
	matrix* Inputs,
	matrix* Intermediate,
	matrix* Result,
	uint32_t Start,
	uint32_t Stride
);
HOST_PREFIX DEVICE_PREFIX
float MseForwardCore(
	matrix* Predictions, matrix* Labels, uint32_t Start, uint32_t Stride
);
HOST_PREFIX DEVICE_PREFIX
float XentropyForwardCore(
	matrix* Predictions, matrix* Labels, uint32_t Start, uint32_t Stride
);
HOST_PREFIX DEVICE_PREFIX
void CreateMiniBatch(
	int_shuffler* IntShuffler,
	matrix* MiniBatchData,
	matrix* MiniBatchLabels,
	matrix* Inputs,
	matrix* Labels,
	uint32_t BatchIndex,
	uint32_t MiniBatchSize,
	uint32_t Start,
	uint32_t Stride
);
void SaveNeuralNet(neural_net* NeuralNet, char* FilePath);

#define NEURAL_NET_H
#endif