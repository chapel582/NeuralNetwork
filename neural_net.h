#ifndef NEURAL_NET_H

#include "matrix.h"

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

struct neural_net
{
	uint32_t NumLayers;
	uint32_t BatchSize;
	uint32_t InputDim;
	layer_link* FirstLink;
	layer_link* LastLink;

	// NOTE: op jobs if needed 
	void* MatrixOpJobs;
};

struct neural_net_trainer
{
	neural_net* NeuralNet;
	void** TrainDataArray;
	matrix* MiniBatchData;
	matrix* MiniBatchLabels;
};

void InitDenseLayers(neural_net* NeuralNet);

#define NEURAL_NET_H
#endif