#ifndef NEURAL_NET_CPU_H

#include "matrix.h"
#include "neural_net.h"

void NeuralNetForward(
	neural_net* NeuralNet,
	matrix* Inputs,
	matrix* Labels,
	matrix** Predictions,
	float* LossResult
);
float TopOneAccuracy(neural_net* NeuralNet, matrix* Inputs, matrix* Labels);

#define NEURAL_NET_CPU_H
#endif
