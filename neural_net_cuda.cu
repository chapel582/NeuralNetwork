#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "neural_net_common.h"

void AllocCudaVector(vector** Result, uint64_t VectorLength)
{
	cudaMallocManaged(Result, sizeof(vector) + (VectorLength * sizeof(float)));
	vector* Vector = *Result;
	Vector->Length = VectorLength;
	Vector->Data = (float*) (((uint8_t*) Vector) + sizeof(vector));
}

void AllocCudaVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array** Result
)
{
	cudaMallocManaged(Result, sizeof(vector_array));
	vector_array* VectorArray = *Result;
	VectorArray->Length = NumVectors;
	cudaMallocManaged(
		&VectorArray->Vectors, VectorArray->Length * sizeof(vector)
	);
	for(int VectorIndex = 0; VectorIndex < VectorArray->Length; VectorIndex++)
	{
		vector* Vector = &VectorArray->Vectors[VectorIndex];
		Vector->Length = VectorLength;
		cudaMallocManaged(&Vector->Data, Vector->Length * sizeof(float));
	}
}

void AllocLayerOutput(
	uint64_t NumInputs, dense_layer Layer, vector_array** Result
)
{
	AllocCudaVectorArray(NumInputs, Layer.Biases.Length, Result);
}

void MakeDenseLayer(
	int InputDim, int OutputDim, dense_layer** Result
)
{
	cudaMallocManaged(Result, sizeof(dense_layer));
	dense_layer* DenseLayer = *Result;

	weights* Weights = &DenseLayer->Weights;
	Weights->Length = OutputDim;
	cudaMallocManaged(&Weights->Vectors, Weights->Length * sizeof(vector));

	for(int VectorIndex = 0; VectorIndex < Weights->Length; VectorIndex++)
	{
		vector* Vector = &Weights->Vectors[VectorIndex];
		Vector->Length = InputDim;
		cudaMallocManaged(&Vector->Data, Vector->Length * sizeof(float));
	}
	// TODO: we might want to randomly initialize the weights to values 
	// CONT: between -1 and 1

	vector* Biases = &DenseLayer->Biases;
	Biases->Length = OutputDim;
	cudaMallocManaged(&Biases->Data, Biases->Length * sizeof(float));
}

// TODO: free memory functions

// TODO: add some in dense forward
__global__
void DenseForward(
	vector_array Inputs, dense_layer DenseLayer, vector_array* Outputs
)
{
	// NOTE: we can probably save on copies here
	weights Weights = DenseLayer.Weights;
	vector Biases = DenseLayer.Biases;

	// NOTE: this basically indexes by the thread index, offset by the block #
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;
	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		// NOTE: we might be able to save on a copy here
		vector Input = Inputs.Vectors[Row];
		vector* Output = &Outputs->Vectors[Row];
		for(int WeightIndex = 0; WeightIndex < Weights.Length; WeightIndex++)
		{
			float DotResult = 0.0;
			vector* WeightVector = &Weights.Vectors[WeightIndex];
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
			Output->Data[WeightIndex] = DotResult + Biases.Data[WeightIndex]; 
		}
	}
}

__global__
void ReluForward(vector_array Inputs, vector_array* Outputs)
{
	// NOTE: this basically indexes by the thread index, offset by the block #
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;
	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		vector Input = Inputs.Vectors[Row];
		vector* Output = &Outputs->Vectors[Row];
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

__global__
void SigmoidForward(vector_array Inputs, vector_array* Outputs)
{
	// NOTE: this basically indexes by the thread index, offset by the block #
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;
	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		vector Input = Inputs.Vectors[Row];
		vector* Output = &Outputs->Vectors[Row];
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
	float Input1Data[4] = {1, 2, 3, 2.5};
	float Input2Data[4] = {2.0f, 5.0f, -1.0f, 2.0f};
	float Weights1Data[4] = {0.2f, 0.8f, -0.5f, 1.0f};
	float Weights2Data[4] = {0.5f, -0.91f, 0.26f, -0.5f};
	float Weights3Data[4] = {-0.26f, -0.27f, 0.17f, 0.87f};
	float BiasData[3] = {2.0f, 3.0f, 0.5f};

	vector_array* Inputs = NULL;
	AllocCudaVectorArray(2, ARRAY_COUNT(Input1Data), &Inputs);
	memcpy(
		Inputs->Vectors[0].Data,
		&Input1Data[0],
		Inputs->Vectors[0].Length * sizeof(float)
	);
	memcpy(
		Inputs->Vectors[1].Data,
		&Input2Data[0],
		Inputs->Vectors[1].Length * sizeof(float)
	);

	dense_layer* Layer1 = NULL;
	MakeDenseLayer(ARRAY_COUNT(Input1Data), ARRAY_COUNT(BiasData), &Layer1);
	weights* Weights = &Layer1->Weights;
	memcpy(
		Weights->Vectors[0].Data,
		&Weights1Data[0],
		Weights->Vectors[0].Length * sizeof(float)
	);
	memcpy(
		Weights->Vectors[1].Data,
		&Weights2Data[0],
		Weights->Vectors[1].Length * sizeof(float)
	);
	memcpy(
		Weights->Vectors[2].Data,
		&Weights3Data[0],
		Weights->Vectors[2].Length * sizeof(float)
	);
	vector* Biases = &Layer1->Biases;
	memcpy(Biases->Data, &BiasData[0], Biases->Length * sizeof(float));

	float Layer2Weights1Data[3] = {1, 0, 0};
	float Layer2Weights2Data[3] = {0, -1, 0};
	float Layer2Weights3Data[3] = {0, 0, 0.5};
	dense_layer* Layer2 = NULL;
	MakeDenseLayer(ARRAY_COUNT(BiasData), ARRAY_COUNT(BiasData), &Layer2);
	Weights = &Layer2->Weights;
	memcpy(
		Weights->Vectors[0].Data,
		&Layer2Weights1Data[0],
		Weights->Vectors[0].Length * sizeof(float)
	);
	memcpy(
		Weights->Vectors[1].Data,
		&Layer2Weights2Data[0],
		Weights->Vectors[1].Length * sizeof(float)
	);
	memcpy(
		Weights->Vectors[2].Data,
		&Layer2Weights3Data[0],
		Weights->Vectors[2].Length * sizeof(float)
	);
	Biases = &Layer2->Biases;
	memset(Biases->Data, 0, sizeof(float) * Biases->Length);

	vector_array* Layer1Outputs = NULL;
	AllocLayerOutput(Inputs->Length, *Layer1, &Layer1Outputs);
	vector_array* Layer2Outputs = NULL;
	AllocLayerOutput(Inputs->Length, *Layer2, &Layer2Outputs);

	// NOTE: ThreadCount can't exceed the number of weights to process
	int BlockSize = 256;
	// NOTE: this is always at least one, and grows as the data to process grows
	int NumBlocks = (Inputs->Length + BlockSize - 1) / BlockSize;

	DenseForward<<<NumBlocks, BlockSize>>>(*Inputs, *Layer1, Layer1Outputs);
	cudaDeviceSynchronize();
	PrintVectorArray(*Layer1Outputs);

	DenseForward<<<NumBlocks, BlockSize>>>(
		*Layer1Outputs, *Layer2, Layer2Outputs
	);
	cudaDeviceSynchronize();
	PrintVectorArray(*Layer2Outputs);

	ReluForward<<<NumBlocks, BlockSize>>>(*Layer2Outputs, Layer2Outputs);
	cudaDeviceSynchronize();
	PrintVectorArray(*Layer2Outputs);

	SigmoidForward<<<NumBlocks, BlockSize>>>(*Layer2Outputs, Layer2Outputs);
	cudaDeviceSynchronize();
	PrintVectorArray(*Layer2Outputs);
	return 0;
}