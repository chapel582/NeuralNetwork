#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "neural_net_common.h"

// TODO: finish CPU threading for matrix operations
// TODO: figure out good error handling and thread-safe assertion scheme 

void DenseForward(
	vector_array Inputs, weights Weights, vector Biases, vector_array* Outputs 
)
{
	// NOTE: this isn't too different from a matrix multiplication
	// CONT: but I don't have to transpose anything
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

int main(void)
{
	float InputsData[4] = {1, 2, 3, 2.5};
	float Weights1Data[4] = {0.2f, 0.8f, -0.5f, 1.0f};
	float Weights2Data[4] = {0.5f, -0.91f, 0.26f, -0.5f};
	float Weights3Data[4] = {-0.26f, -0.27f, 0.17f, 0.87f};
	float BiasData[3] = {2.0f, 3.0f, 0.5f};
	vector Biases = {};
	Biases.Length = ARRAY_COUNT(BiasData);
	Biases.Data = &BiasData[0];

	vector_array Inputs = {};
	Inputs.Length = 1;
	Inputs.Vectors = (vector*) malloc(sizeof(vector) * Inputs.Length);
	Inputs.Vectors[0].Length = 4;
	Inputs.Vectors[0].Data = &InputsData[0];

	vector WeightsData[3];
	weights Weights = {};
	Weights.Length = ARRAY_COUNT(WeightsData);
	Weights.Vectors = &WeightsData[0];
	Weights.Vectors[0].Length = 4;
	Weights.Vectors[0].Data = &Weights1Data[0];
	Weights.Vectors[1].Length = 4;
	Weights.Vectors[1].Data = &Weights2Data[0];
	Weights.Vectors[2].Length = 4;
	Weights.Vectors[2].Data = &Weights3Data[0];

	vector_array Outputs = {}; 
	Outputs.Length = Inputs.Length;
	Outputs.Vectors = (vector*) malloc(sizeof(vector) * Outputs.Length);
	for(int OutputIndex = 0; OutputIndex < Outputs.Length; OutputIndex++)
	{
		Outputs.Vectors[OutputIndex].Length = Weights.Length;
		Outputs.Vectors[OutputIndex].Data = (
			(float*) malloc(sizeof(float) * Outputs.Vectors[0].Length)
		);
	}
	DenseForward(Inputs, Weights, Biases, &Outputs);
	
	PrintVectorArray(Outputs);
	return 0;
}