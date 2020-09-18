#ifndef NEURAL_NET_H

#include "project_flags.h"
#include "matrix.h"

// TODO: Need to have a platform independent way of handling threads
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

typedef enum
{
	LayerType_Dense,
	LayerType_Relu,
	LayerType_Softmax,
	LayerType_CrossEntropy,
	LayerType_SoftmaxCrossEntropy,
	LayerType_Mse,
	LayerType_Count
} layer_type;

struct layer_link;
struct layer_link
{
	layer_type Type;
	void* Data;
	matrix* Output;
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

void InitDenseLayers(neural_net* NeuralNet);
void AllocMatrixOpJobs(matrix_op_jobs** Result, uint32_t NumThreads);
float MeanSquaredForward(
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

#define NEURAL_NET_H
#endif