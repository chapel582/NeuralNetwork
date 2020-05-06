#include <stdio.h>
#include <stdint.h>

#include "neural_net_common.h"

void InitCudaVector(vector** Result, int VectorLength)
{
	cudaMallocManaged(Result, sizeof(vector) + VectorLength * sizeof(float));
	vector* Vector = *Result;
	Vector->Length = VectorLength;
	Vector->Data = (float*) (((uint8_t*) Vector) + sizeof(vector));
}

__global__
void PropagateForward(
	vector Input, weights Weights, vector Biases, vector* Output
)
{
	// NOTE: this basically indexes by the thread index, offset by the block #
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;
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
			DotResult += (
				Input.Data[VectorIndex] * WeightVector->Data[VectorIndex]
			);
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
		Weights->Vectors[0].Length * sizeof(float)
	);
	Weights->Vectors[1].Length = 4;
	Weights->Vectors[1].Data = (
		Weights->Vectors[0].Data + Weights->Vectors[0].Length
	);
	memcpy(
		Weights->Vectors[1].Data,
		&Weights2Data[0],
		Weights->Vectors[1].Length * sizeof(float)
	);
	Weights->Vectors[2].Length = 4;
	Weights->Vectors[2].Data = (
		Weights->Vectors[1].Data + Weights->Vectors[1].Length
	);
	memcpy(
		Weights->Vectors[2].Data,
		&Weights3Data[0],
		Weights->Vectors[2].Length * sizeof(float)
	);

	vector* NextLayer = NULL;
	InitCudaVector(&NextLayer, Weights->Length);

	// NOTE: ThreadCount can't exceed the number of weights to process
	int BlockSize = 256;
	// NOTE: this is always at least one, and grows as the data to process grows
	int NumBlocks = (Weights->Length + BlockSize - 1) / BlockSize;
	PropagateForward<<<NumBlocks, BlockSize>>>(
		*Inputs, *Weights, *Biases, NextLayer
	);
	cudaDeviceSynchronize();
	
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