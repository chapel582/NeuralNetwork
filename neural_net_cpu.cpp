#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neural_net_common.h"

void AllocVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array* Result
)
{
	// NOTE: not in common b/c it's in CPU memory, not shared memory with GPU
	// NOTE: sets vector array values to 0 by default

	Result->Length = NumVectors;
	
	// NOTE: we can potentially make this faster by making just one call to 
	// CONT: malloc
	Result->Vectors = (vector*) malloc(Result->Length * sizeof(vector));
	for(int ElementIndex = 0; ElementIndex < Result->Length; ElementIndex++)
	{
		Result->Vectors[ElementIndex].Length = VectorLength;
		size_t VectorSize = (
			sizeof(float) * Result->Vectors[ElementIndex].Length
		);
		Result->Vectors[ElementIndex].Data = (float*) malloc(VectorSize);
		memset(Result->Vectors[ElementIndex].Data, 0, VectorSize);
	}
}

inline void AllocVectorArraySameDim(
	vector_array CopyDimFrom, vector_array* Result
)
{
	return AllocVectorArray(
		CopyDimFrom.Length, CopyDimFrom.Vectors[0].Length, Result
	);
}

void AllocLayerOutput(
	uint64_t NumInputs, dense_layer Layer, vector_array* Result
)
{
	AllocVectorArray(NumInputs, Layer.Biases.Length, Result);
}

// TODO: finish CPU threading for matrix operations
// TODO: figure out good error handling and thread-safe assertion scheme 
void MakeDenseLayer(
	int InputDim, int OutputDim, dense_layer* DenseLayer
)
{
	weights* Weights = &DenseLayer->Weights;
	Weights->Length = OutputDim;

	Weights->Vectors = (vector*) malloc(Weights->Length * sizeof(vector));
	for(int VectorIndex = 0; VectorIndex < Weights->Length; VectorIndex++)
	{
		Weights->Vectors[VectorIndex].Length = InputDim;
		Weights->Vectors[VectorIndex].Data = (
			(float*) malloc(
				Weights->Vectors[VectorIndex].Length * sizeof(float)
			)
		);
	}
	// TODO: we might want to randomly initialize the weights to values 
	// CONT: between -1 and 1

	vector* Biases = &DenseLayer->Biases;
	Biases->Length = OutputDim;
	Biases->Data = (float*) malloc(Biases->Length * sizeof(float));
	// TODO: we might want to initialize Biases->Data to be slightly greater 
	// CONT: than zero by default to avoid dead networks
}

void DenseForward(
	vector_array Inputs, dense_layer DenseLayer, vector_array* Outputs 
)
{
	// NOTE: this isn't too different from a matrix multiplication
	// CONT: but I don't have to transpose anything
	weights Weights = DenseLayer.Weights;
	vector Biases = DenseLayer.Biases;

	ASSERT(Inputs.Length == Outputs->Length);
	ASSERT(Biases.Length == Weights.Length);
	for(int Row = 0; Row < Inputs.Length; Row++)
	{
		vector Input = Inputs.Vectors[Row];
		vector* Output = &Outputs->Vectors[Row];
		ASSERT(Output->Length == Weights.Length);
		ASSERT(Output->Length == Biases.Length);
		for(
			int WeightIndex = 0;
			WeightIndex < Weights.Length;
			WeightIndex++
		)
		{
			vector Weight = Weights.Vectors[WeightIndex];
			ASSERT(Weight.Length == Input.Length);
			Output->Data[WeightIndex] = 0.0f;
			// NOTE: dot product between input and weights
			for(
				int VectorIndex = 0;
				VectorIndex < Input.Length;
				VectorIndex++
			)
			{
				Output->Data[WeightIndex] += (
					(Input.Data[VectorIndex] * Weight.Data[VectorIndex])
				);
			}
			// NOTE: add bias
			Output->Data[WeightIndex] += Biases.Data[WeightIndex];
		}
	}
}

void ReluForward(vector_array Inputs, vector_array* Outputs)
{
	ASSERT(Inputs.Length == Outputs->Length);
	for(int Row = 0; Row < Inputs.Length; Row++)
	{
		vector Input = Inputs.Vectors[Row];
		vector* Output = &Outputs->Vectors[Row];
		ASSERT(Input.Length == Output->Length);
		for(int ElementIndex = 0; ElementIndex < Input.Length; ElementIndex++)
		{
			if(Input.Data[ElementIndex] < 0)
			{
				Output->Data[ElementIndex] = 0;
			}
			else
			{
				Output->Data[ElementIndex] = Input.Data[ElementIndex];
			}
		}
	}
}

void SigmoidForward(vector_array Inputs, vector_array* Outputs)
{
	ASSERT(Inputs.Length == Outputs->Length);
	for(int Row = 0; Row < Inputs.Length; Row++)
	{
		vector Input = Inputs.Vectors[Row];
		vector* Output = &Outputs->Vectors[Row];
		ASSERT(Input.Length == Output->Length);
		for(int ElementIndex = 0; ElementIndex < Input.Length; ElementIndex++)
		{
			Output->Data[ElementIndex] = (float) (
				1.0f / (1 + exp(-1 * Input.Data[ElementIndex]))
			);
		}
	}
}

int main(void)
{
	/*
	TODO: remove this silly temp data result reminder
	[4.800000, 1.210000, 2.385000]
	[8.900000, -1.810000, 0.200000]

	[4.800000, -1.210000, 1.192500]
	[8.900000, 1.810000, 0.100000]

	[4.800000, 0.000000, 1.192500]
	[8.900000, 1.810000, 0.100000]
	*/
	float Input1Data[4] = {1, 2, 3, 2.5};
	float Input2Data[4] = {2.0f, 5.0f, -1.0f, 2.0f};
	float Weights1Data[4] = {0.2f, 0.8f, -0.5f, 1.0f};
	float Weights2Data[4] = {0.5f, -0.91f, 0.26f, -0.5f};
	float Weights3Data[4] = {-0.26f, -0.27f, 0.17f, 0.87f};
	float BiasData[3] = {2.0f, 3.0f, 0.5f};

	vector_array Inputs = {};
	AllocVectorArray(2, ARRAY_COUNT(Input1Data), &Inputs);
	memcpy(
		Inputs.Vectors[0].Data,
		&Input1Data[0],
		Inputs.Vectors[0].Length * sizeof(float)
	);
	memcpy(
		Inputs.Vectors[1].Data,
		&Input2Data[0],
		Inputs.Vectors[1].Length * sizeof(float)
	);

	dense_layer Layer1 = {};
	MakeDenseLayer(ARRAY_COUNT(Input1Data), ARRAY_COUNT(BiasData), &Layer1);
	weights* Weights = &Layer1.Weights;
	memcpy(
		Weights->Vectors[0].Data,
		&Weights1Data[0],
		sizeof(float) * Weights->Vectors[0].Length
	);
	memcpy(
		Weights->Vectors[1].Data,
		&Weights2Data[0],
		sizeof(float) * Weights->Vectors[1].Length
	);
	memcpy(
		Weights->Vectors[2].Data,
		&Weights3Data[0],
		sizeof(float) * Weights->Vectors[2].Length
	);
	vector* Biases = &Layer1.Biases;
	memcpy(Biases->Data, &BiasData, sizeof(float) * Biases->Length);

	float Layer2Weights1Data[3] = {1, 0, 0};
	float Layer2Weights2Data[3] = {0, -1, 0};
	float Layer2Weights3Data[3] = {0, 0, 0.5};
	dense_layer Layer2 = {};
	MakeDenseLayer(ARRAY_COUNT(BiasData), ARRAY_COUNT(BiasData), &Layer2);
	Weights = &Layer2.Weights;
	memcpy(
		Weights->Vectors[0].Data,
		&Layer2Weights1Data[0],
		sizeof(float) * Weights->Vectors[0].Length
	);
	memcpy(
		Weights->Vectors[1].Data,
		&Layer2Weights2Data[0],
		sizeof(float) * Weights->Vectors[1].Length
	);
	memcpy(
		Weights->Vectors[2].Data,
		&Layer2Weights3Data[0],
		sizeof(float) * Weights->Vectors[2].Length
	);
	Biases = &Layer2.Biases;
	memset(Biases->Data, 0, sizeof(float) * Biases->Length);

	vector_array Layer1Outputs = {};
	AllocLayerOutput(Inputs.Length, Layer1, &Layer1Outputs);
	vector_array Layer2Outputs = {};	
	AllocLayerOutput(Inputs.Length, Layer2, &Layer2Outputs);

	DenseForward(Inputs, Layer1, &Layer1Outputs);
	PrintVectorArray(Layer1Outputs);
	DenseForward(Layer1Outputs, Layer2, &Layer2Outputs);
	PrintVectorArray(Layer2Outputs);
	ReluForward(Layer2Outputs, &Layer2Outputs);
	PrintVectorArray(Layer2Outputs);
	SigmoidForward(Layer2Outputs, &Layer2Outputs);
	PrintVectorArray(Layer2Outputs);
	
	return 0;
}