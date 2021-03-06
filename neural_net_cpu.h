#ifndef NEURAL_NET_CPU_H

#include "project_flags.h"
#include "matrix.h"
#include "neural_net.h"

float TopOneAccuracy(neural_net* NeuralNet, matrix* Inputs, matrix* Labels);

#define NEURAL_NET_CPU_H
#endif
