#include <stdio.h>
#include <stdint.h>

#define ARRAY_COUNT(Array) (sizeof(Array) / sizeof(Array[0]))
#define ASSERT(Expression) if(!(Expression)) {*(int*) 0 = 0;}

struct vector
{
	uint64_t Length;
	float* Data;	
};

void InitCudaVector(vector** Result, int VectorLength)
{
	cudaMallocManaged(Result, sizeof(vector) + VectorLength * sizeof(float));
	vector* Vector = *Result;
	Vector->Length = VectorLength;
	Vector->Data = (float*) (((uint8_t*) Vector) + sizeof(vector));
}

struct weights
{
	uint64_t Length;
	vector* Vectors;
};

__global__
void PropagateForward(
	vector Input, weights Weights, vector Biases, vector* Output
)
{
	ASSERT(
		(Weights.Length == Output->Length) && 
		(Weights.Length == Biases.Length) && 
		(Output->Length == Biases.Length)
	);
	int Start = 0; //blockIdx.x * blockDim.x + threadIdx.x;
	int Stride = 1; //blockDim.x * gridDim.x;
	for(int Index = Start; Index < Weights.Length; Index += Stride)
	{
		float DotResult = 0.0;
		vector* WeightVector = &Weights.Vectors[Index];
		for(
			int VectorIndex = 0;
			VectorIndex < WeightVector->Length;
			VectorIndex++
		)
		{
			DotResult += Input.Data[Index] * WeightVector->Data[Index];
		}
		// TODO: parallelize vector addition?
		Output->Data[Index] = DotResult + Biases.Data[Index]; 
	}
}

int main(void)
{
	float InputsData[4] = {1, 2, 3, 2.5};
	float Weights1Data[4] = {0.2f, 0.8f, -0.5f, 1.0f};
	float Weights2Data[4] = {0.5f, -0.91f, 0.26f, -0.5f};
	float Weights3Data[4] = {-0.26f, -0.27f, 0.17f, 0.87f};
	float BiasData[3] = {2.0f, 3.0f, 0.5f};
	
	vector* Biases = NULL;
	InitCudaVector(&Biases, ARRAY_COUNT(BiasData));
	memcpy(Biases->Data, &BiasData[0], Biases->Length * sizeof(float));

	vector* Inputs = NULL;
	InitCudaVector(&Inputs, ARRAY_COUNT(InputsData));
	memcpy(Inputs->Data, &InputsData[0], Inputs->Length * sizeof(float));

	weights* Weights = NULL;
	size_t VectorsArraySize = 3 * sizeof(vector); // NOTE: 3 is a magic number right now. it's the number of weights 
	int VectorDataCount = ARRAY_COUNT(Weights1Data);
	cudaMallocManaged(
		&Weights,
		(
			sizeof(weights) + 
			VectorsArraySize + 
			3 * VectorDataCount * sizeof(float)
		)
	);
	Weights->Length = 3;
	Weights->Vectors = (vector*) (((uint8_t*) Weights) + sizeof(weights));
	float* StartOfVectorData = (
		(float*) (((uint8_t*) Weights->Vectors) + VectorsArraySize)
	);
	Weights->Vectors[0].Length = 4;
	Weights->Vectors[0].Data = StartOfVectorData;
	memcpy(
		Weights->Vectors[0].Data,
		&Weights1Data[0],
		Weights->Vectors[0].Length
	);
	Weights->Vectors[1].Length = 4;
	Weights->Vectors[1].Data = (
		Weights->Vectors[0].Data + Weights->Vectors[0].Length
	);
	memcpy(
		Weights->Vectors[1].Data,
		&Weights2Data[0],
		Weights->Vectors[1].Length
	);
	Weights->Vectors[2].Length = 4;
	Weights->Vectors[2].Data = (
		Weights->Vectors[1].Data + Weights->Vectors[1].Length
	);
	memcpy(
		Weights->Vectors[2].Data,
		&Weights3Data[0],
		Weights->Vectors[2].Length
	);

	vector* NextLayer = NULL;
	InitCudaVector(&NextLayer, Weights->Length);
	PropagateForward<<<1, 1>>>(*Inputs, *Weights, *Biases, NextLayer);
	
	// TODO: pull this out into print array
	printf("[");
	int Index;
	for(Index = 0; Index < (NextLayer->Length - 1); Index++)
	{
		printf("%f, ", NextLayer->Data[Index]);
	}
	printf("%f]\n", NextLayer->Data[Index]);
	return 0;
}