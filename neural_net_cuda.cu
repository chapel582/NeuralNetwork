#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "neural_net_common.h"

__device__
inline float* CudaGetFirstVector(vector_array VectorArray)
{
	return VectorArray.Vectors;
}

__device__
inline float* CudaGetVector(vector_array VectorArray, uint64_t Index)
{
	return VectorArray.Vectors + Index * VectorArray.VectorLength;
}

void AllocCudaVector(uint64_t Length, vector** Result)
{
	cudaMallocManaged(Result, sizeof(vector));
	vector* Vector = *Result;
	Vector->Length = Length;
	cudaMallocManaged(&Vector->Data, GetVectorDataSize(*Vector));
}

void AllocCudaVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array** Result
)
{
	cudaMallocManaged(Result, sizeof(vector_array));
	vector_array* VectorArray = *Result;
	VectorArray->Length = NumVectors;
	VectorArray->VectorLength = VectorLength;
	cudaMallocManaged(
		&VectorArray->Vectors, GetVectorArrayDataSize(*VectorArray)
	);
}

void MakeCudaVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array** Result
)
{
	AllocCudaVectorArray(NumVectors, VectorLength, Result);
	vector_array* VectorArray = *Result;
	memset(
		VectorArray->Vectors, 0, GetVectorArrayDataSize(*VectorArray)
	);
}

inline void AllocCudaVectorArraySameDim(
	vector_array CopyDimFrom, vector_array** Result
)
{
	return AllocCudaVectorArray(
		CopyDimFrom.Length, CopyDimFrom.VectorLength, Result
	);
}

void AllocCudaLayerOutput(
	uint64_t NumInputs, dense_layer Layer, vector_array** Result
)
{
	AllocCudaVectorArray(NumInputs, Layer.Biases->Length, Result);
}

void AllocCudaDenseLayer(
	uint64_t InputDim, uint64_t OutputDim, dense_layer** Result
)
{
	cudaMallocManaged(Result, sizeof(dense_layer));
	dense_layer* DenseLayer = *Result;	
	
	AllocCudaVectorArray(OutputDim, InputDim, &DenseLayer->Weights);
	// TODO: we might want to randomly initialize the weights to values 
	// CONT: between -1 and 1 in a "make" function

	AllocCudaVector(OutputDim, &DenseLayer->Biases);
	// TODO: we might want to initialize Biases->Data to be slightly greater 
	// CONT: than zero by default to avoid dead networks in a "make" function
}

__global__
void CudaDenseForward(
	vector_array Inputs, dense_layer DenseLayer, vector_array* Outputs
)
{
	// NOTE: we can probably save on copies here
	weights* Weights = DenseLayer.Weights;
	vector* Biases = DenseLayer.Biases;
	float* BiasData = Biases->Data;

	// NOTE: this basically indexes by the thread index, offset by the block #
	int Start = blockIdx.x * blockDim.x + threadIdx.x;  
	// NOTE: this basically calculates the # of threads
	int Stride = blockDim.x * gridDim.x;
	for(int Row = Start; Row < Inputs.Length; Row += Stride)
	{
		// NOTE: we might be able to save on a copy here
		float* Input = CudaGetVector(Inputs, Row);
		float* Output = CudaGetVector(*Outputs, Row);
		for(int WeightIndex = 0; WeightIndex < Weights->Length; WeightIndex++)
		{
			float DotResult = 0.0;
			float* WeightVector = CudaGetVector(*Weights, WeightIndex);
			for(
				int ElementIndex = 0;
				ElementIndex < Weights->VectorLength;
				ElementIndex++
			)
			{
				DotResult += (
					Input[ElementIndex] * WeightVector[ElementIndex]
				);
			}
			Output[WeightIndex] = DotResult + BiasData[WeightIndex]; 
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
		float* Input = CudaGetVector(Inputs, Row);
		float* Output = CudaGetVector(*Outputs, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			if(Input[ElementIndex] < 0)
			{
				Output[ElementIndex] = 0;
			}
			else
			{
				Output[ElementIndex] = Input[ElementIndex];
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
		float* Input = CudaGetVector(Inputs, Row);
		float* Output = CudaGetVector(*Outputs, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			Output[ElementIndex] = (float) (
				1.0f / (1 + exp(-1 * Input[ElementIndex]))
			);
		}
	}
}

void MakeCudaSpiralData(
	int PointsPerClass,
	int Dimensions,
	int NumClasses,
	vector_array** InputsResults,
	vector_array** OutputsResults
)
{
	// NOTE: this is for testing only
	// NOTE: based on https://cs231n.github.io/neural-networks-case-study/
	// NOTE: data comes normalized already
	AllocCudaVectorArray(
		NumClasses * PointsPerClass, Dimensions, InputsResults
	);
	MakeCudaVectorArray(
		NumClasses * PointsPerClass, NumClasses, OutputsResults
	);
	InitSpiralData(
		PointsPerClass,
		Dimensions,
		NumClasses,
		*InputsResults,
		*OutputsResults
	);
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

	[0.991837, 0.500000, 0.767188]
	[0.999864, 0.859362, 0.524979]
	*/
	float Input1Data[4] = {1, 2, 3, 2.5};
	float Input2Data[4] = {2.0f, 5.0f, -1.0f, 2.0f};
	float Weights1Data[4] = {0.2f, 0.8f, -0.5f, 1.0f};
	float Weights2Data[4] = {0.5f, -0.91f, 0.26f, -0.5f};
	float Weights3Data[4] = {-0.26f, -0.27f, 0.17f, 0.87f};
	float BiasData[3] = {2.0f, 3.0f, 0.5f};

	vector_array* Inputs = NULL;
	AllocCudaVectorArray(2, ARRAY_COUNT(Input1Data), &Inputs);
	memcpy(GetVector(*Inputs, 0), &Input1Data[0], GetVectorDataSize(*Inputs));
	memcpy(GetVector(*Inputs, 1), &Input2Data[0], GetVectorDataSize(*Inputs));

	dense_layer* Layer1 = NULL;
	AllocCudaDenseLayer(
		ARRAY_COUNT(Input1Data), ARRAY_COUNT(BiasData), &Layer1
	);
	weights* Weights = Layer1->Weights;
	memcpy(
		GetVector(*Weights, 0),
		&Weights1Data[0],
		GetVectorDataSize(*Weights)
	);
	memcpy(
		GetVector(*Weights, 1),
		&Weights2Data[0],
		GetVectorDataSize(*Weights)
	);
	memcpy(
		GetVector(*Weights, 2),
		&Weights3Data[0],
		GetVectorDataSize(*Weights)
	);
	vector* Biases = Layer1->Biases;
	memcpy(Biases->Data, &BiasData[0], GetVectorDataSize(*Biases));

	float Layer2Weights1Data[3] = {1, 0, 0};
	float Layer2Weights2Data[3] = {0, -1, 0};
	float Layer2Weights3Data[3] = {0, 0, 0.5};
	dense_layer* Layer2 = NULL;
	AllocCudaDenseLayer(ARRAY_COUNT(BiasData), ARRAY_COUNT(BiasData), &Layer2);
	Weights = Layer2->Weights;
	memcpy(
		GetVector(*Weights, 0),
		&Layer2Weights1Data[0],
		GetVectorDataSize(*Weights)
	);
	memcpy(
		GetVector(*Weights, 1),
		&Layer2Weights2Data[0],
		GetVectorDataSize(*Weights)
	);
	memcpy(
		GetVector(*Weights, 2),
		&Layer2Weights3Data[0],
		GetVectorDataSize(*Weights)
	);
	Biases = Layer2->Biases;
	memset(Biases->Data, 0, sizeof(float) * Biases->Length);

	vector_array* Layer1Outputs = NULL;
	AllocCudaLayerOutput(Inputs->Length, *Layer1, &Layer1Outputs);
	vector_array* Layer2Outputs = NULL;	
	AllocCudaLayerOutput(Inputs->Length, *Layer2, &Layer2Outputs);

	int BlockSize = 256;
	// NOTE: this is always at least one, and grows as the data to process grows
	int NumBlocks = (Inputs->Length + BlockSize - 1) / BlockSize;
	CudaDenseForward<<<NumBlocks, BlockSize>>>(*Inputs, *Layer1, Layer1Outputs);
	cudaError_t SyncResult = cudaDeviceSynchronize();
	ASSERT(SyncResult == cudaSuccess);
	PrintVectorArray(*Layer1Outputs);
	CudaDenseForward<<<NumBlocks, BlockSize>>>(
		*Layer1Outputs, *Layer2, Layer2Outputs
	);
	SyncResult = cudaDeviceSynchronize();
	ASSERT(SyncResult == cudaSuccess);
	PrintVectorArray(*Layer2Outputs);
	ReluForward<<<NumBlocks, BlockSize>>>(*Layer2Outputs, Layer2Outputs);
	SyncResult = cudaDeviceSynchronize();
	ASSERT(SyncResult == cudaSuccess);	
	PrintVectorArray(*Layer2Outputs);
	SigmoidForward<<<NumBlocks, BlockSize>>>(*Layer2Outputs, Layer2Outputs);
	SyncResult = cudaDeviceSynchronize();
	ASSERT(SyncResult == cudaSuccess);	
	PrintVectorArray(*Layer2Outputs);

	printf("\n");
	vector_array* SpiralInputs = NULL;
	vector_array* SpiralOutputs = NULL;
	int PointsPerClass = 100;
	int NumClasses = 3;
	int Dimensions = 2;
	MakeCudaSpiralData(
		PointsPerClass, Dimensions, NumClasses, &SpiralInputs, &SpiralOutputs
	);
	for(int ClassIndex = 0; ClassIndex < NumClasses; ClassIndex++)
	{
		for(int DataIndex = 0; DataIndex < SpiralInputs->Length; DataIndex++)
		{
			float* SpiralOutput = GetVector(*SpiralOutputs, DataIndex);
			int Classification = ArgMax(
				SpiralOutput, SpiralOutputs->VectorLength
			);
			if(Classification == ClassIndex)
			{
				float* SpiralInput = GetVector(*SpiralInputs, DataIndex); 
				for(
					int ElementIndex = 0;
					ElementIndex < Dimensions;
					ElementIndex++
				)
				{
					printf("%f,", SpiralInput[ElementIndex]);
				}
				printf("%d", Classification);
				printf("\n");
			}
		}
		printf("\n\n");
	}
	return 0;
}