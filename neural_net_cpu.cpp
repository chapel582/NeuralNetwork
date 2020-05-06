#include <stdio.h>
#include <stdint.h>

#include "neural_net_common.h"

float Dot(vector V1, vector V2)
{
	// TODO: should this be an assertion?
	ASSERT(V1.Length == V2.Length);
	float Result = 0.0;
	for(int Index = 0; Index < V1.Length; Index++)
	{
		Result += V1.Data[Index] * V2.Data[Index];
	}
	return Result;
}

void PropagateForward(
	vector Input, weights Weights, vector Biases, vector* Output
)
{
	ASSERT(
		(Weights.Length == Output->Length) && 
		(Weights.Length == Biases.Length) && 
		(Output->Length == Biases.Length)
	);
	for(int Index = 0; Index < Weights.Length; Index++)
	{
		Output->Data[Index] = (
			Dot(Input, Weights.Vectors[Index]) + Biases.Data[Index]
		); 
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

	vector Inputs = {};
	Inputs.Length = 4;
	Inputs.Data = &InputsData[0];

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

	float NextLayerData[3] = {0.0, 0.0, 0.0};
	vector NextLayer = {};
	NextLayer.Length = ARRAY_COUNT(NextLayerData);
	NextLayer.Data = &NextLayerData[0];
	PropagateForward(Inputs, Weights, Biases, &NextLayer);
	
	// TODO: pull this out into print array
	printf("[");
	int Index;
	for(Index = 0; Index < (NextLayer.Length - 1); Index++)
	{
		printf("%f, ", NextLayer.Data[Index]);
	}
	printf("%f]\n", NextLayer.Data[Index]);
	return 0;
}