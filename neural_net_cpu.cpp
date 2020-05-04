#include <stdio.h>
#include <stdint.h>

#define ARRAY_COUNT(Array) (sizeof(Array) / sizeof(Array[0]))
#define ASSERT(Expression) if(!(Expression)) {*(int*) 0 = 0;}
struct vector
{
	uint64_t Length;
	float* Data;	
};

// TODO: operator overloading for accessing element of vector?

float Dot(vector V1, vector V2)
{
	// TODO: should we use pointers to the structs since these arrays are 
	// CONT: arbitrarily large?
	// TODO: should this be an assertion?
	ASSERT(V1.Length == V2.Length);
	float Result = 0.0;
	for(int Index = 0; Index < V1.Length; Index++)
	{
		Result += V1.Data[Index] * V2.Data[Index];
	}
	return Result;
}

int main(void)
{
	float InputsData[4] = {1, 2, 3, 2.5};
	float Weights1Data[4] = {0.2f, 0.8f, -0.5f, 1.0f};
	float Weights2Data[4] = {0.5f, -0.91f, 0.26f, -0.5f};
	float Weights3Data[4] = {-0.26f, -0.27f, 0.17f, 0.87f};
	float Bias1 = 2.0f;
	float Bias2 = 3.0f;
	float Bias3 = 0.5f;

	vector Inputs = {};
	Inputs.Length = 4;
	Inputs.Data = &InputsData[0];

	vector Weights1 = {};
	Weights1.Length = 4;
	Weights1.Data = &Weights1Data[0];
	vector Weights2 = {};
	Weights2.Length = 4;
	Weights2.Data = &Weights2Data[0];
	vector Weights3 = {};
	Weights3.Length = 4;
	Weights3.Data = &Weights3Data[0];

	float NextLayerData[3] = {0.0, 0.0, 0.0};
	vector NextLayer = {};
	NextLayer.Length = ARRAY_COUNT(NextLayerData);
	NextLayer.Data = &NextLayerData[0];
	NextLayer.Data[0] = Dot(Inputs, Weights1) + Bias1;
	NextLayer.Data[1] = Dot(Inputs, Weights2) + Bias2;
	NextLayer.Data[2] = Dot(Inputs, Weights3) + Bias3;
	
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