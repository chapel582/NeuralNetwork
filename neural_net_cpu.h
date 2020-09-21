#ifndef NEURAL_NET_CPU_H

#include "project_flags.h"
#include "matrix.h"
#include "neural_net.h"

float TopOneAccuracy(neural_net* NeuralNet, matrix* Inputs, matrix* Labels);
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

#define NEURAL_NET_CPU_H
#endif
