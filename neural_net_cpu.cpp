#include <stdio.h>
#include <stdint.h>

#define ARRAY_COUNT(Array) (sizeof(Array) / sizeof(Array[0]))
#define ASSERT(Expression) if(!(Expression)) {*(int*) 0 = 0;}
struct float_array
{
	uint64_t Length;
	float* Data;
};

struct vector
{
	uint64_t Length;
	float* Data;	
};

// TODO: operator overloading for accessing element of vector?

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

struct weights
{
	uint64_t Length;
	vector* Data;
};

void PropagateForward(
	vector Input, weights Weights, float_array Biases, vector* Output
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
			Dot(Input, Weights.Data[Index]) + Biases.Data[Index]
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
	float_array Biases = {};
	Biases.Length = ARRAY_COUNT(BiasData);
	Biases.Data = &BiasData[0];

	vector Inputs = {};
	Inputs.Length = 4;
	Inputs.Data = &InputsData[0];

	vector WeightsData[3];
	weights Weights = {};
	Weights.Length = ARRAY_COUNT(WeightsData);
	Weights.Data = &WeightsData[0];
	Weights.Data[0].Length = 4;
	Weights.Data[0].Data = &Weights1Data[0];
	Weights.Data[1].Length = 4;
	Weights.Data[1].Data = &Weights2Data[0];
	Weights.Data[2].Length = 4;
	Weights.Data[2].Data = &Weights3Data[0];

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