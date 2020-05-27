#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// TODO: Need to have a platform independent way of handling threads
#include <windows.h>

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
	
	AllocVector(OutputDim, &DenseLayer->Biases);
}

void MakeDenseLayer(
	uint64_t InputDim, uint64_t OutputDim, dense_layer** Result
)
{
	AllocDenseLayer(InputDim, OutputDim, Result);
	InitDenseLayer(*Result);
}

void FreeDenseLayer(dense_layer* DenseLayer)
{
	FreeVector(DenseLayer->Biases);
	FreeVectorArray(DenseLayer->Weights);
	free(DenseLayer);
}

struct thread_dense_forward_args
{
	vector_array Inputs;
	dense_layer DenseLayer;
	vector_array* Outputs;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadDenseForward(void* VoidArgs)
{
	thread_dense_forward_args* Args = (thread_dense_forward_args*) VoidArgs;
	vector_array Inputs = Args->Inputs;
	dense_layer DenseLayer = Args->DenseLayer;
	vector_array* Outputs = Args->Outputs;
	weights* Weights = DenseLayer.Weights;
	vector* Biases = DenseLayer.Biases;
	float* BiasData = Biases->Data;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Inputs.Length; Row += Stride)
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
	return 0;
}

void DenseForward(
	vector_array Inputs,
	dense_layer DenseLayer,
	vector_array* Outputs,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_dense_forward_args* ThreadArgs
)
{
	// NOTE: only works for windows right now
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_dense_forward_args* Args = &ThreadArgs[Index];
		Args->Inputs = Inputs;
		Args->DenseLayer = DenseLayer;
		Args->Outputs = Outputs;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadDenseForward, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
}

struct thread_relu_forward_args
{
	vector_array Inputs;
	vector_array* Outputs;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadReluForward(void* VoidArgs)
{
	thread_relu_forward_args* Args = (thread_relu_forward_args*) VoidArgs;
	vector_array Inputs = Args->Inputs;
	vector_array* Outputs = Args->Outputs;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Inputs.Length; Row += Stride)
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

	return 0;
}

void ReluForward(
	vector_array Inputs,
	vector_array* Outputs,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_relu_forward_args* ThreadArgs
)
{
	// NOTE: only works for windows right now
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_relu_forward_args* Args = &ThreadArgs[Index];
		Args->Inputs = Inputs;
		Args->Outputs = Outputs;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadReluForward, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
}

struct thread_sigmoid_forward_args
{
	vector_array Inputs;
	vector_array* Outputs;
	int Start;
	int Stride;
};

DWORD WINAPI ThreadSigmoidForward(void* VoidArgs)
{
	thread_sigmoid_forward_args* Args = (thread_sigmoid_forward_args*) VoidArgs;
	vector_array Inputs = Args->Inputs;
	vector_array* Outputs = Args->Outputs;
	int Start = Args->Start;
	int Stride = Args->Stride;

	for(int Row = Start; Row < Inputs.Length; Row += Stride)
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
	
	return 0;
}

void SigmoidForward(
	vector_array Inputs,
	vector_array* Outputs,
	HANDLE* ThreadHandles,
	int NumThreads,
	thread_sigmoid_forward_args* ThreadArgs
)
{
	// NOTE: only works for windows right now
	for(int Index = 0; Index < NumThreads; Index++)
	{
		DWORD ThreadId;
		thread_sigmoid_forward_args* Args = &ThreadArgs[Index];
		Args->Inputs = Inputs;
		Args->Outputs = Outputs;
		Args->Start = Index;
		Args->Stride = NumThreads;
		ThreadHandles[Index] = CreateThread( 
			NULL, // default security attributes
			0, // use default stack size  
			ThreadSigmoidForward, // thread function name
			Args, // argument to thread function 
			0, // use default creation flags 
			&ThreadId // returns the thread identifier
		);
	}
	WaitForMultipleObjects(NumThreads, ThreadHandles, TRUE, INFINITE);
}

void SoftmaxForward(vector_array Inputs, vector_array* Outputs)
{
	for(int Row = 0; Row < Inputs.Length; Row++)
	{
		float Sum = 0;
		float* Input = GetVector(Inputs, Row);
		float* Output = GetVector(*Outputs, Row);
		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			Input[ElementIndex] = (float) exp(Input[ElementIndex]);
			Sum += Input[ElementIndex];
		}

		for(
			int ElementIndex = 0;
			ElementIndex < Inputs.VectorLength;
			ElementIndex++
		)
		{
			Output[ElementIndex] = Input[ElementIndex] / Sum;
		}
	}
}

void MakeSpiralData(
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
	AllocVectorArray(NumClasses * PointsPerClass, Dimensions, InputsResults);
	MakeVectorArray(NumClasses * PointsPerClass, NumClasses, OutputsResults);
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

	[
	[0.659001, 0.242433, 0.098566]
	]
	[
	[0.576117, 0.211942, 0.211942]
	]
	*/

	// NOTE: set up thread stuff so we don't have to do it on every layer of
	// NOTE: the forward pass
	// TODO: turn this into a function
	int NumThreads = 4;
	HANDLE* ThreadHandles = (HANDLE*) malloc(NumThreads * sizeof(HANDLE)); 
	thread_dense_forward_args* DenseForwardArgs = (thread_dense_forward_args*) (
		malloc(NumThreads * sizeof(thread_dense_forward_args))
	);
	thread_relu_forward_args* ReluForwardArgs = (thread_relu_forward_args*) (
		malloc(NumThreads * sizeof(thread_relu_forward_args))
	);
	thread_sigmoid_forward_args* SigmoidForwardArgs = (
		(thread_sigmoid_forward_args*) 
		malloc(NumThreads * sizeof(thread_sigmoid_forward_args))
	);

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
	memcpy(Biases->Data, &BiasData[0], GetVectorDataSize(*Biases));

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

	DenseForward(
		*Inputs,
		*Layer1,
		Layer1Outputs,
		ThreadHandles,
		NumThreads,
		DenseForwardArgs
	);
	PrintVectorArray(*Layer1Outputs);
	DenseForward(
		*Layer1Outputs,
		*Layer2,
		Layer2Outputs,
		ThreadHandles,
		NumThreads,
		DenseForwardArgs
	);
	PrintVectorArray(*Layer2Outputs);
	ReluForward(
		*Layer2Outputs,
		Layer2Outputs,
		ThreadHandles,
		NumThreads,
		ReluForwardArgs
	);
	PrintVectorArray(*Layer2Outputs);
	SigmoidForward(
		*Layer2Outputs,
		Layer2Outputs,
		ThreadHandles,
		NumThreads,
		SigmoidForwardArgs
	);
	PrintVectorArray(*Layer2Outputs);
	printf("\n");

	printf("RandInit test\n");
	dense_layer* RandInitLayer = NULL;
	MakeDenseLayer(
		ARRAY_COUNT(Input1Data), ARRAY_COUNT(BiasData), &RandInitLayer
	);
	vector_array* Layer3Outputs = NULL;
	AllocLayerOutput(Inputs->Length, *RandInitLayer, &Layer3Outputs);
	DenseForward(
		*Inputs,
		*RandInitLayer,
		Layer3Outputs,
		ThreadHandles,
		NumThreads,
		DenseForwardArgs
	);
	PrintVectorArray(*Layer3Outputs);
	printf("\n");

	printf("SoftmaxForward test\n");
	float SoftmaxData[3] = {2.0f, 1.0f, 0.1f};
	float SoftmaxData2[3] = {1.0f, 0.0f, 0.0f};
	vector_array* SoftmaxForwardInputs = NULL;
	AllocVectorArray(2, 3, &SoftmaxForwardInputs);
	memcpy(
		GetVector(*SoftmaxForwardInputs, 0),
		&SoftmaxData,
		GetVectorDataSize(*SoftmaxForwardInputs)
	);
	memcpy(
		GetVector(*SoftmaxForwardInputs, 1),
		&SoftmaxData2,
		GetVectorDataSize(*SoftmaxForwardInputs)
	);
	SoftmaxForward(*SoftmaxForwardInputs, SoftmaxForwardInputs);
	PrintVectorArray(*SoftmaxForwardInputs);
	printf("\n");

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
	if(Layer3Outputs)
	{
		FreeVectorArray(Layer3Outputs);
	}

#if 0
	printf("\n");
	vector_array* SpiralInputs = NULL;
	vector_array* SpiralOutputs = NULL;
	int PointsPerClass = 100;
	int NumClasses = 3;
	int Dimensions = 2;
	MakeSpiralData(
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
#endif 
	return 0;
}