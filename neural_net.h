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

#endif