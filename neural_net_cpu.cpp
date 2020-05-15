#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neural_net_common.h"

void AllocVector(uint64_t Length, vector** Result)
{
	*Result = (vector*) malloc(sizeof(vector));
	vector* Vector = *Result;
	Vector->Length = Length;
	Vector->Data = (float*) malloc(GetVectorDataSize(*Vector));
}

void FreeVector(vector* Vector)
{
	free(Vector->Data);
	free(Vector);
}

void AllocVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array** Result
)
{
	// NOTE: not in common b/c it's in CPU memory, not shared memory with GPU
	*Result = (vector_array*) malloc(sizeof(vector_array));
	vector_array* VectorArray = *Result;
	VectorArray->Length = NumVectors;
	VectorArray->VectorLength = VectorLength;
	VectorArray->Vectors = (float*) malloc(
		GetVectorArrayDataSize(*VectorArray)
	);
}

void FreeVectorArray(vector_array* VectorArray)
{
	free(VectorArray->Vectors);
	free(VectorArray);
}

void MakeVectorArray(
	uint64_t NumVectors, uint64_t VectorLength, vector_array** Result
)
{
	AllocVectorArray(NumVectors, VectorLength, Result);
	vector_array* VectorArray = *Result;
	memset(VectorArray->Vectors, 0, GetVectorArrayDataSize(*VectorArray));
}

inline void AllocVectorArraySameDim(
	vector_array CopyDimFrom, vector_array** Result
)
{
	return AllocVectorArray(
		CopyDimFrom.Length, CopyDimFrom.VectorLength, Result
	);
}

void AllocLayerOutput(
	uint64_t NumInputs, dense_layer Layer, vector_array** Result
)
{
	AllocVectorArray(NumInputs, Layer.Biases->Length, Result);
}

// TODO: finish CPU threading for matrix operations
// TODO: figure out good error handling and thread-safe assertion scheme 
void AllocDenseLayer(
	uint64_t InputDim, uint64_t OutputDim, dense_layer** Result
)
{
	*Result = (dense_layer*) malloc(sizeof(dense_layer));
	dense_layer* DenseLayer = *Result;	
	
	AllocVectorArray(OutputDim, InputDim, &DenseLayer->Weights);
	// TODO: we might want to randomly initialize the weights to values 
	// CONT: between -1 and 1 in a "make" function

	AllocVector(OutputDim, &DenseLayer->Biases);
	// TODO: we might want to initialize Biases->Data to be slightly greater 
	// CONT: than zero by default to avoid dead networks in a "make" function
}

void FreeDenseLayer(dense_layer* DenseLayer)
{
	FreeVector(DenseLayer->Biases);
	FreeVectorArray(DenseLayer->Weights);
	free(DenseLayer);
}

void DenseForward(
	vector_array Inputs, dense_layer DenseLayer, vector_array* Outputs
)
{
	// NOTE: we can probably save on copies here
	weights* Weights = DenseLayer.Weights;
	vector* Biases = DenseLayer.Biases;
	float* BiasData = Biases->Data;

	for(int Row = 0; Row < Inputs.Length; Row++)
	{
		// NOTE: we might be able to save on a copy here
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
		for(int WeightIndex = 0; WeightIndex < Weights->Length; WeightIndex++)
		{
			float DotResult = 0.0;
			float* WeightVector = GetVector(*Weights, WeightIndex);
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

void ReluForward(vector_array Inputs, vector_array* Outputs)
{
	for(int Row = 0; Row < Inputs.Length; Row++)
	{
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
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

void SigmoidForward(vector_array Inputs, vector_array* Outputs)
{
	for(int Row = 0; Row < Inputs.Length; Row++)
	{
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
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
	dense_layer* Layer1 = NULL;
	dense_layer* Layer2 = NULL;
	vector_array* Layer1Outputs = NULL;
	vector_array* Layer2Outputs = NULL;	

	AllocVectorArray(2, ARRAY_COUNT(Input1Data), &Inputs);
	memcpy(GetVector(*Inputs, 0), &Input1Data[0], GetVectorDataSize(*Inputs));
	memcpy(GetVector(*Inputs, 1), &Input2Data[0], GetVectorDataSize(*Inputs));

	AllocDenseLayer(ARRAY_COUNT(Input1Data), ARRAY_COUNT(BiasData), &Layer1);
	weights* Weights = Layer1->Weights;
	memcpy(
		GetVector(*Weights, 0), &Weights1Data[0], GetVectorDataSize(*Weights)
	);
	memcpy(
		GetVector(*Weights, 1), &Weights2Data[0], GetVectorDataSize(*Weights)
	);
	memcpy(
		GetVector(*Weights, 2), &Weights3Data[0], GetVectorDataSize(*Weights)
	);

	vector* Biases = Layer1->Biases;
	memcpy(Biases->Data, &BiasData[0], Biases->Length);

	float Layer2Weights1Data[3] = {1, 0, 0};
	float Layer2Weights2Data[3] = {0, -1, 0};
	float Layer2Weights3Data[3] = {0, 0, 0.5};
	AllocDenseLayer(ARRAY_COUNT(BiasData), ARRAY_COUNT(BiasData), &Layer2);
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
	memset(Biases->Data, 0, GetVectorDataSize(*Biases));

	AllocLayerOutput(Inputs->Length, *Layer1, &Layer1Outputs);
	AllocLayerOutput(Inputs->Length, *Layer2, &Layer2Outputs);

	DenseForward(*Inputs, *Layer1, Layer1Outputs);
	PrintVectorArray(*Layer1Outputs);
	DenseForward(*Layer1Outputs, *Layer2, Layer2Outputs);
	PrintVectorArray(*Layer2Outputs);
	ReluForward(*Layer2Outputs, Layer2Outputs);
	PrintVectorArray(*Layer2Outputs);
	SigmoidForward(*Layer2Outputs, Layer2Outputs);
	PrintVectorArray(*Layer2Outputs);

	// NOTE: for this test code, we don't need to free, but it's here if we need
	// CONT: it and we might as well test it 
	if(Inputs)
	{
		FreeVectorArray(Inputs);		
	}
	if(Layer1)
	{
		FreeDenseLayer(Layer1);
	}
	if(Layer2)
	{
		FreeDenseLayer(Layer2);
	}	
	if(Layer1Outputs)
	{
		FreeVectorArray(Layer1Outputs);
	}
	if(Layer2Outputs)
	{
		FreeVectorArray(Layer2Outputs);	
	}
	return 0;
}